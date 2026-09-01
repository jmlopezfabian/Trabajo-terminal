"""
Regenera la tabla de cobertura por municipio.

Este archivo (``ntl_data/municipios_coordenadas_pixeles.json``) precalcula,
una sola vez por municipio, qué fracción de cada píxel de la retícula de Black Marble
cae dentro de su límite geográfico. El pipeline de radianza lo lee en cada
ejecución diaria para saltarse por completo la etapa de transformación
geométrica.

Guarda tripletas (x, y, w) con w en (0, 1], calculadas por intersección
geométrica exacta. La versión anterior guardaba solo coordenadas, lo que
obligaba a decidir cada píxel de frontera entero: como esas celdas están
cubiertas aproximadamente por la mitad, descartarlas subestimaba el área del
municipio entre 7% y 31% según su forma.

La geometría del recorte depende únicamente del tamaño de la retícula del
producto y de la esquina superior izquierda del cuadrante, ambos derivables del
identificador del cuadrante (hHHvVV). Por eso no hace falta descargar ningún
HDF5 para regenerar el archivo.

Uso:
    python scripts/generar_coordenadas_pixeles.py [--salida ruta.json] [--dry-run]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntl.core.config import RUTA_MUNICIPIOS
from ntl.geometria.cobertura import cobertura_exacta, poligono_en_pixeles
from ntl.core.utils import (
    cuadrante_de_coordenadas,
    esquina_superior_izquierda,
    extraer_coordenadas,
    normalize_municipio,
)

# Retícula de Black Marble a 500 m: 2400x2400 píxeles por cuadrante de 10°x10°
FORMA_CUADRANTE = (2400, 2400)

SALIDA_POR_DEFECTO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ntl_data",
    "municipios_coordenadas_pixeles.json",
)


def cobertura_de_municipio(nombre: str, cuadrante: str) -> list[list]:
    """
    Devuelve las tripletas (x, y, w) del municipio en la retícula del cuadrante.

    La cobertura se calcula por intersección geométrica exacta: sin factor de
    subdivisión y sin ningún peso que elegir para el trazo del borde. La
    aproximación por subdivisión sigue disponible en `pesos_municipio`, pero no
    hay motivo para usarla aquí, donde el cálculo se paga una sola vez.
    """
    coordenadas = extraer_coordenadas(nombre)
    if coordenadas is None:
        raise ValueError(f"No se encontraron coordenadas para el municipio: {nombre}")

    upper_left = esquina_superior_izquierda(cuadrante)
    poligono = poligono_en_pixeles(coordenadas, upper_left, FORMA_CUADRANTE)
    pesos, fila_0, columna_0 = cobertura_exacta(poligono)

    # Una esquina del polígono puede rozar una celda y dejar una cobertura de
    # 1e-8; al redondear queda en 0.0 y sobra, porque no aporta área ni valor.
    filas, columnas = np.nonzero(pesos)
    tripletas = [
        [int(x) + columna_0, int(y) + fila_0, round(float(pesos[y, x]), 6)]
        for y, x in zip(filas, columnas)
    ]
    return [t for t in tripletas if t[2] > 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", default=SALIDA_POR_DEFECTO,
                        help="Ruta del JSON a escribir (por defecto, el del paquete)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo compara contra el archivo existente, sin escribir")
    args = parser.parse_args()

    previo = {}
    if os.path.exists(args.salida):
        with open(args.salida, "r", encoding="utf-8") as f:
            previo = json.load(f)

    with open(RUTA_MUNICIPIOS, "r", encoding="utf-8") as f:
        municipios = json.load(f)["features"]

    resultado = {}
    for feature in municipios:
        nombre = feature["properties"]["NOMGEO"]
        clave = normalize_municipio(nombre)

        # El cuadrante se deduce del propio polígono. Antes se heredaba del
        # archivo previo, así que un municipio nuevo se omitía en silencio y la
        # tabla no podía crecer sin editarla a mano.
        try:
            cuadrante = cuadrante_de_coordenadas(extraer_coordenadas(nombre))
        except ValueError as e:
            print(f"AVISO: {nombre} se omite: {e}")
            continue

        heredado = previo.get(clave, {}).get("cuadrante")
        if heredado and heredado != cuadrante:
            print(f"  AVISO: {nombre} estaba en {heredado} y sus coordenadas "
                  f"caen en {cuadrante}")

        pesos = cobertura_de_municipio(nombre, cuadrante)
        resultado[clave] = {
            "nombre": clave,
            "cuadrante": cuadrante,
            "pesos": pesos,
        }

        area = sum(w for _, _, w in pesos)
        frontera = sum(1 for _, _, w in pesos if w < 0.999)
        anterior = previo.get(clave, {})
        entradas_previas = anterior.get("pesos") or anterior.get("coordenadas_pixeles", [])
        area_previa = (
            sum(p[2] for p in entradas_previas) if entradas_previas and len(entradas_previas[0]) == 3
            else len(entradas_previas)
        )
        cambio = (area / area_previa - 1) * 100 if area_previa else float("nan")
        print(f"{nombre:26s} {cuadrante}  area {area:8.2f} px  "
              f"({frontera:4d} de frontera)   antes {area_previa:8.2f}  {cambio:+6.1f}%")

        # Qué se considera un cambio aceptable depende de con qué se compara.
        # Frente a una tabla de coordenadas, el área solo puede crecer: la nueva
        # incluye la frontera. Frente a una tabla que ya trae coberturas, debe
        # coincidir salvo por el error de la aproximación anterior, que a k=32
        # era de milésimas.
        era_cobertura = bool(entradas_previas) and len(entradas_previas[0]) == 3
        if area_previa:
            if era_cobertura:
                if abs(area / area_previa - 1) > 0.01:
                    print(f"  ERROR: el área de {nombre} cambia más de 1%: "
                          f"{area_previa:.2f} -> {area:.2f}")
                    return 1
            elif area < area_previa:
                print(f"  ERROR: el área de {nombre} baja de {area_previa:.2f} a {area:.2f}")
                return 1

        # Un píxel puede desaparecer si el método anterior lo incluía por error.
        # El relleno viejo llegaba a marcar como interiores bolsas que el
        # polígono cierra sin llegar a tocarlas.
        anteriores = {(p[0], p[1]) for p in entradas_previas}
        perdidos = sorted(anteriores - {(x, y) for x, y, _ in pesos})
        if perdidos:
            print(f"  AVISO: {len(perdidos)} píxeles del archivo previo tienen cobertura "
                  f"nula y salen: {perdidos[:5]}{' ...' if len(perdidos) > 5 else ''}")

    if args.dry_run:
        print("\n--dry-run: no se escribió nada.")
        return 0

    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f)
    print(f"\nEscrito: {args.salida} ({len(resultado)} municipios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
