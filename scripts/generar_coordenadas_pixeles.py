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
from ntl.geometria.mosaico import cobertura_por_cuadrante
from ntl.core.utils import (
    extraer_geometria,
    normalize_municipio,
)

# Retícula de Black Marble a 500 m: 2400x2400 píxeles por cuadrante de 10°x10°
FORMA_CUADRANTE = (2400, 2400)

SALIDA_POR_DEFECTO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ntl_data",
    "municipios_coordenadas_pixeles.json",
)


def cobertura_de_municipio(nombre: str) -> dict[str, list[list]]:
    """
    Devuelve la cobertura del municipio repartida por cuadrante: {cuadrante: [(x, y, w)]}.

    La cobertura se calcula por intersección geométrica sobre la retícula
    global, sin factor de subdivisión y sin ningún peso que elegir para el trazo
    del borde. El cuadrante deja de ser un dato del municipio y pasa a ser una
    consecuencia de dónde caen sus píxeles: un municipio a caballo entre dos
    imágenes produce dos piezas en vez de un error.

    Se parte de la geometría completa, con sus islas y sus huecos. Un enclave de
    otro municipio dentro del territorio resta área, y una isla la suma aunque
    caiga en otro cuadrante.
    """
    geometria = extraer_geometria(nombre)
    if geometria is None:
        raise ValueError(f"No se encontraron coordenadas para el municipio: {nombre}")

    piezas = cobertura_por_cuadrante(geometria, FORMA_CUADRANTE)
    return {c: [[int(x), int(y), w] for x, y, w in pesos] for c, pesos in piezas.items()}


def entrada_de_tabla(clave: str, piezas: dict[str, list[list]]) -> dict:
    """
    Registro que se escribe en el JSON.

    Con un solo cuadrante conserva la forma de siempre —``cuadrante`` y
    ``pesos``— para no reescribir de arriba abajo un archivo que solo cambia de
    contenedor; con varios usa ``piezas``. El modelo lee las dos.
    """
    if len(piezas) == 1:
        (cuadrante, pesos), = piezas.items()
        return {"nombre": clave, "cuadrante": cuadrante, "pesos": pesos}
    return {
        "nombre": clave,
        "piezas": [{"cuadrante": c, "pesos": p} for c, p in piezas.items()],
    }


def area_previa_de(anterior: dict) -> tuple[float, bool, set]:
    """Área, si traía coberturas, y píxeles (cuadrante, x, y) del registro previo."""
    entradas = []
    if anterior.get("piezas"):
        for pieza in anterior["piezas"]:
            entradas += [(pieza["cuadrante"], *p) for p in pieza["pesos"]]
    else:
        cuadrante = anterior.get("cuadrante")
        for p in anterior.get("pesos") or anterior.get("coordenadas_pixeles", []):
            entradas.append((cuadrante, *p))

    if not entradas:
        return 0.0, False, set()
    era_cobertura = len(entradas[0]) == 4
    area = sum(e[3] for e in entradas) if era_cobertura else float(len(entradas))
    return area, era_cobertura, {(e[0], e[1], e[2]) for e in entradas}


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

        # Los cuadrantes se deducen del propio polígono. Antes se heredaba el
        # cuadrante del archivo previo, así que un municipio nuevo se omitía en
        # silencio y la tabla no podía crecer sin editarla a mano; y un
        # municipio que cruzaba de cuadrante se omitía siempre.
        try:
            piezas = cobertura_de_municipio(nombre)
        except ValueError as e:
            print(f"AVISO: {nombre} se omite: {e}")
            continue

        if not piezas:
            print(f"AVISO: {nombre} se omite: no cubre ningún píxel de la retícula")
            continue

        anterior = previo.get(clave, {})
        heredados = anterior.get("cuadrante")
        if heredados and [heredados] != list(piezas):
            print(f"  AVISO: {nombre} estaba en {heredados} y sus píxeles caen "
                  f"en {', '.join(piezas)}")

        resultado[clave] = entrada_de_tabla(clave, piezas)

        area = sum(w for pesos in piezas.values() for _, _, w in pesos)
        frontera = sum(1 for pesos in piezas.values() for _, _, w in pesos if w < 0.999)
        area_previa, era_cobertura, anteriores = area_previa_de(anterior)
        cambio = (area / area_previa - 1) * 100 if area_previa else float("nan")
        etiqueta = ", ".join(piezas) if len(piezas) > 1 else next(iter(piezas))
        print(f"{nombre:26s} {etiqueta:26s}  area {area:8.2f} px  "
              f"({frontera:4d} de frontera)   antes {area_previa:8.2f}  {cambio:+6.1f}%")

        # Qué se considera un cambio aceptable depende de con qué se compara.
        # Frente a una tabla de coordenadas, el área solo puede crecer: la nueva
        # incluye la frontera. Frente a una tabla que ya trae coberturas, debe
        # coincidir salvo por el error de la aproximación anterior, que a k=32
        # era de milésimas.
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
        actuales = {(c, x, y) for c, pesos in piezas.items() for x, y, _ in pesos}
        perdidos = sorted(anteriores - actuales)
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
