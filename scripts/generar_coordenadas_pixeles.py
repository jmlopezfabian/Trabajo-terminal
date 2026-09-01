"""
Regenera la tabla de cobertura por municipio.

Este archivo (``vnp46a1_data/municipios_coordenadas_pixeles.json``) precalcula,
una sola vez por municipio, qué fracción de cada píxel de la retícula VNP46A1
cae dentro de su límite geográfico. El pipeline de radianza lo lee en cada
ejecución diaria para saltarse por completo la etapa de transformación
geométrica.

Guarda tripletas (x, y, w) con w en (0, 1]. La versión anterior guardaba solo
coordenadas, lo que obligaba a decidir cada píxel de frontera entero: como esas
celdas están cubiertas aproximadamente por la mitad, descartarlas subestimaba el
área del municipio entre 7% y 31% según su forma.

La geometría del recorte depende únicamente del tamaño de la retícula del
producto y de la esquina superior izquierda del cuadrante, ambos derivables del
identificador del cuadrante (hHHvVV). Por eso no hace falta descargar ningún
HDF5 para regenerar el archivo.

Uso:
    python scripts/generar_coordenadas_pixeles.py [--salida ruta.json] [--factor N] [--dry-run]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vnp46a1.core.config import RUTA_MUNICIPIOS
from vnp46a1.geometria.image_processor import pesos_municipio, recortar
from vnp46a1.core.utils import extraer_coordenadas, normalize_municipio

# Retícula del producto VNP46A1 a 500 m: 2400x2400 píxeles por cuadrante de 10°x10°
FORMA_CUADRANTE = (2400, 2400)

SALIDA_POR_DEFECTO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vnp46a1_data",
    "municipios_coordenadas_pixeles.json",
)


def esquina_superior_izquierda(cuadrante: str) -> tuple[float, float]:
    """Longitud y latitud de la esquina superior izquierda de un cuadrante hHHvVV."""
    h = int(cuadrante[1:3])
    v = int(cuadrante[4:6])
    return (-180.0 + 10.0 * h, 90.0 - 10.0 * v)


def cobertura_de_municipio(nombre: str, cuadrante: str, factor: int) -> list[list]:
    """
    Devuelve las tripletas (x, y, w) del municipio en la retícula del cuadrante.

    Se recorta al bounding box, se calcula la cobertura de cada píxel sobre una
    malla `factor` veces más fina y se devuelven las coordenadas al sistema
    absoluto del cuadrante sumando el desplazamiento del recorte.
    """
    coordenadas = extraer_coordenadas(nombre)
    if coordenadas is None:
        raise ValueError(f"No se encontraron coordenadas para el municipio: {nombre}")

    upper_left = esquina_superior_izquierda(cuadrante)

    # recortar solo usa la forma de la matriz, no sus valores
    matriz_vacia = np.zeros(FORMA_CUADRANTE, dtype=np.float32)
    recorte, nuevos_x, nuevos_y = recortar(matriz_vacia, coordenadas, upper_left, factor)
    pesos = pesos_municipio(recorte.shape, nuevos_x, nuevos_y, factor)

    # Mismo desplazamiento que aplica recortar al definir el área de recorte
    resolucion_x = 10 / FORMA_CUADRANTE[1]
    resolucion_y = 10 / FORMA_CUADRANTE[0]
    x_pixels = (coordenadas[:, 0] - upper_left[0]) / resolucion_x
    y_pixels = (upper_left[1] - coordenadas[:, 1]) / resolucion_y
    desplazamiento_x = int(np.ceil(x_pixels.min())) - 1
    desplazamiento_y = int(np.ceil(y_pixels.min())) - 1

    filas, columnas = np.nonzero(pesos)
    return [
        [int(x) + desplazamiento_x, int(y) + desplazamiento_y, round(float(pesos[y, x]), 6)]
        for y, x in zip(filas, columnas)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", default=SALIDA_POR_DEFECTO,
                        help="Ruta del JSON a escribir (por defecto, el del paquete)")
    parser.add_argument("--factor", type=int, default=32,
                        help="Subdivisiones por lado al medir la cobertura (por defecto 32). "
                             "El costo es de milisegundos y solo se paga una vez.")
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

        if clave not in previo:
            print(f"AVISO: {nombre} no está en el archivo previo; se omite "
                  f"(no se conoce su cuadrante)")
            continue

        cuadrante = previo[clave]["cuadrante"]
        pesos = cobertura_de_municipio(nombre, cuadrante, args.factor)
        resultado[clave] = {
            "nombre": clave,
            "cuadrante": cuadrante,
            "pesos": pesos,
        }

        area = sum(w for _, _, w in pesos)
        frontera = sum(1 for _, _, w in pesos if w < 0.999)
        entradas_previas = previo[clave].get("pesos") or previo[clave].get("coordenadas_pixeles", [])
        area_previa = (
            sum(p[2] for p in entradas_previas) if entradas_previas and len(entradas_previas[0]) == 3
            else len(entradas_previas)
        )
        cambio = (area / area_previa - 1) * 100 if area_previa else float("nan")
        print(f"{nombre:26s} {cuadrante}  area {area:8.2f} px  "
              f"({frontera:4d} de frontera)   antes {area_previa:8.2f}  {cambio:+6.1f}%")

        # El área es el invariante, no la identidad de cada píxel: el conjunto
        # nuevo suma la frontera, así que no puede encoger.
        if area_previa and area < area_previa:
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
