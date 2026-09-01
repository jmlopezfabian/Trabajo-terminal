"""
Cuánto cambian las métricas municipales si la retícula está desalineada.

La tabla de coberturas se construye sobre un origen derivado del identificador
del cuadrante. `ntl.core.utils.verificar_georreferencia` comprueba que ese
supuesto coincide con lo que declara el archivo, pero no dice cuánto importaría
que no coincidiera. Este script lo mide: recalcula las métricas leyendo la
radianza con la retícula corrida un píxel y compara contra la corrida real.

Es un análisis, no una prueba: necesita un HDF5 de verdad, así que vive en
scripts/ y no en tests/. Las pruebas que sí corren en CI validan el rasterizado
(oracle de Shapely) y la partición de fronteras, que no dependen de la imagen.

Uso:
    python scripts/sensibilidad_desplazamiento.py ruta/al/VNP46A2.h5 [--cuadrante h08v07]
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntl.core.config import PIXELES_MUNICIPIOS, RUTA_MUNICIPIOS
from ntl.core.lectura import leer_radianza
from ntl.core.metricas import metricas_ponderadas
from ntl.core.utils import normalize_municipio

# Un píxel en cada dirección, más las dos diagonales: suficiente para acotar el
# efecto sin multiplicar la salida. El desalineamiento realista es un
# desplazamiento rígido, no una deformación.
DESPLAZAMIENTOS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]


def metricas_desplazadas(imagen: np.ndarray, pesos: list, dx: int, dy: int) -> dict:
    """Métricas del municipio leyendo la radianza corrida (dx, dy) píxeles."""
    alto, ancho = imagen.shape
    dentro = [
        (x, y, w) for x, y, w in pesos
        if 0 <= y + dy < alto and 0 <= x + dx < ancho
    ]
    valores = np.array([float(imagen[y + dy, x + dx]) for x, y, _ in dentro])
    cobertura = np.array([w for _, _, w in dentro], dtype=float)
    return metricas_ponderadas(valores, cobertura)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", help="Archivo VNP46A2 descargado")
    parser.add_argument("--cuadrante", default=None,
                        help="Solo los municipios de este cuadrante (por defecto, "
                             "el que se deduce del nombre del archivo)")
    parser.add_argument("--metrica", default="Media_de_radianza",
                        help="Métrica a comparar (por defecto, la media ponderada)")
    args = parser.parse_args()

    cuadrante = args.cuadrante
    if cuadrante is None:
        partes = [p for p in os.path.basename(args.h5).split(".") if p.startswith("h") and "v" in p]
        if not partes:
            print("No pude deducir el cuadrante del nombre; usa --cuadrante")
            return 1
        cuadrante = partes[0]

    # Misma lectura que el pipeline: unidades físicas y NaN donde no hay medición.
    with h5py.File(args.h5, "r") as hdf:
        imagen, _ = leer_radianza(hdf)

    with open(PIXELES_MUNICIPIOS, encoding="utf-8") as f:
        tabla = json.load(f)
    with open(RUTA_MUNICIPIOS, encoding="utf-8") as f:
        municipios = json.load(f)["features"]

    filas = []
    for feature in municipios:
        nombre = feature["properties"]["NOMGEO"]
        clave = normalize_municipio(nombre)
        if clave not in tabla or tabla[clave]["cuadrante"] != cuadrante:
            continue

        pesos = [(int(x), int(y), float(w)) for x, y, w in tabla[clave]["pesos"]]
        base = metricas_desplazadas(imagen, pesos, 0, 0)[args.metrica]
        if not base:
            continue
        desviaciones = {
            d: 100.0 * (metricas_desplazadas(imagen, pesos, *d)[args.metrica] / base - 1)
            for d in DESPLAZAMIENTOS
        }
        filas.append((nombre, base, desviaciones))

    if not filas:
        print(f"Ningún municipio de la tabla está en {cuadrante}")
        return 1

    encabezado = " ".join(f"{str(d):>9s}" for d in DESPLAZAMIENTOS)
    print(f"\n{args.metrica} con la retícula corrida (dx, dy) píxeles\n")
    print(f"{'municipio':24s} {'sin correr':>10s} {encabezado} {'|peor|':>7s}")
    for nombre, base, desv in filas:
        peor = max(abs(v) for v in desv.values())
        print(f"{nombre:24s} {base:10.2f} "
              + " ".join(f"{desv[d]:+8.2f}%" for d in DESPLAZAMIENTOS)
              + f" {peor:6.2f}%")

    peores = [max(abs(v) for v in d.values()) for _, _, d in filas]
    peor_municipio = max(filas, key=lambda r: max(abs(v) for v in r[2].values()))[0]
    print(f"\nDesviación de {args.metrica} ante 1 píxel de desalineamiento:")
    print(f"  mediana {np.median(peores):.2f}%   media {np.mean(peores):.2f}%   "
          f"máxima {max(peores):.2f}% ({peor_municipio})")

    # El ranking entre municipios es lo que suele leerse de estos datos; que se
    # mueva es más grave que un sesgo común, que al menos se cancela al comparar.
    orden = [n for n, b, _ in sorted(filas, key=lambda r: -r[1])]
    for d in DESPLAZAMIENTOS:
        movido = [n for n, b, dv in sorted(filas, key=lambda r: -(r[1] * (1 + r[2][d] / 100)))]
        cambian = sum(1 for a, b in zip(orden, movido) if a != b)
        print(f"  corrido {str(d):>8s}: {cambian}/{len(orden)} municipios cambian de posición")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
