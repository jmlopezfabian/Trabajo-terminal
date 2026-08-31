"""
Regenera el archivo de coordenadas de píxeles por municipio.

Este archivo (``vnp46a1_data/municipios_coordenadas_pixeles.json``) precalcula,
una sola vez por municipio, el conjunto de píxeles de la retícula VNP46A1 que
caen dentro de su límite geográfico. El pipeline asíncrono lo lee en cada
ejecución diaria para saltarse por completo la etapa de transformación
geométrica.

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

from satellite_sync.config import RUTA_MUNICIPIOS
from satellite_sync.image_processor import completar_bordes, recortar_imagen, get_pixeles
from satellite_sync.utils import extraer_coordenadas, normalize_municipio

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


def coordenadas_de_municipio(nombre: str, cuadrante: str) -> list[list[int]]:
    """
    Devuelve las coordenadas (x, y) del municipio en la retícula completa del cuadrante.

    Se recorta al bounding box del municipio, se cierra el borde y se toman los
    píxeles interiores; luego se devuelven las coordenadas al sistema absoluto
    del cuadrante sumando el desplazamiento del recorte.
    """
    coordenadas = extraer_coordenadas(nombre)
    if coordenadas is None:
        raise ValueError(f"No se encontraron coordenadas para el municipio: {nombre}")

    upper_left = esquina_superior_izquierda(cuadrante)

    # recortar_imagen solo usa la forma de la matriz, no sus valores
    matriz_vacia = np.zeros(FORMA_CUADRANTE, dtype=np.float32)
    imagen_recortada, nuevos_x, nuevos_y = recortar_imagen(
        matriz_vacia, coordenadas, upper_left, factor_escala=1
    )

    bordes = completar_bordes(nuevos_x, nuevos_y)
    pixeles = get_pixeles(imagen_recortada, bordes)

    # Mismo desplazamiento que aplica recortar_imagen al definir el área de recorte
    resolucion_x = 10 / FORMA_CUADRANTE[1]
    resolucion_y = 10 / FORMA_CUADRANTE[0]
    x_pixels = (coordenadas[:, 0] - upper_left[0]) / resolucion_x
    y_pixels = (upper_left[1] - coordenadas[:, 1]) / resolucion_y
    desplazamiento_x = int(np.ceil(x_pixels.min())) - 1
    desplazamiento_y = int(np.ceil(y_pixels.min())) - 1

    return [[x + desplazamiento_x, y + desplazamiento_y] for x, y in pixeles]


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

        if clave not in previo:
            print(f"AVISO: {nombre} no está en el archivo previo; se omite "
                  f"(no se conoce su cuadrante)")
            continue

        cuadrante = previo[clave]["cuadrante"]
        pixeles = coordenadas_de_municipio(nombre, cuadrante)
        resultado[clave] = {
            "nombre": clave,
            "cuadrante": cuadrante,
            "coordenadas_pixeles": pixeles,
        }

        anteriores = {tuple(p) for p in previo[clave]["coordenadas_pixeles"]}
        actuales = {tuple(p) for p in pixeles}
        perdidos = len(anteriores - actuales)
        nuevos = len(actuales - anteriores)
        marca = "  <-- CAMBIA" if (perdidos or nuevos) else ""
        print(f"{nombre:26s} {cuadrante}  {len(pixeles):5d} px  "
              f"(antes {len(anteriores):5d}, +{nuevos} -{perdidos}){marca}")

        if perdidos:
            print(f"  ERROR: se perderían {perdidos} píxeles de {nombre}")
            return 1

    if args.dry_run:
        print("\n--dry-run: no se escribió nada.")
        return 0

    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f)
    print(f"\nEscrito: {args.salida} ({len(resultado)} municipios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
