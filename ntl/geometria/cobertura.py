"""
Cobertura de un municipio sobre la retícula, por intersección geométrica.

Tras llevar el polígono a coordenadas de píxel, el píxel de índices (j, k) es el
cuadrado unitario [j, j+1] x [k, k+1] y la pertenencia deja de ser una decisión
binaria: es el área de la intersección entre el polígono y ese cuadrado.

Decidir cada píxel de frontera entero —aceptarlo o descartarlo— costaba entre el
7% y el 31% del territorio según la forma del municipio, porque el anillo que la
frontera atraviesa crece con el perímetro mientras el interior crece con el área.
"""
from typing import Tuple

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from ..core.metricas import metricas_ponderadas as _metricas_nucleo

GRADOS_POR_CUADRANTE = 10.0


def poligono_en_pixeles(coordenadas_municipio,
                        upper_left: Tuple[float, float],
                        shape: Tuple[int, int]):
    """
    Lleva el límite del municipio de lon/lat a coordenadas de píxel continuas.

    Acepta una geometría de shapely —con sus islas y sus huecos— o un arreglo de
    vértices, que es como lo pedía el proyecto antes de que un municipio pudiera
    tener más de una parte. La transformación es afín, así que se aplica igual a
    un polígono simple que a uno con veinte islas.
    """
    resolucion_x = GRADOS_POR_CUADRANTE / shape[1]
    resolucion_y = GRADOS_POR_CUADRANTE / shape[0]

    if isinstance(coordenadas_municipio, BaseGeometry):
        # x = (lon - ul_x) / res_x ;  y = (ul_y - lat) / res_y
        return affine_transform(coordenadas_municipio, [
            1 / resolucion_x, 0.0,
            0.0, -1 / resolucion_y,
            -upper_left[0] / resolucion_x, upper_left[1] / resolucion_y,
        ])

    coordenadas_municipio = np.asarray(coordenadas_municipio)
    xs = (coordenadas_municipio[:, 0] - upper_left[0]) / resolucion_x
    ys = (upper_left[1] - coordenadas_municipio[:, 1]) / resolucion_y
    return Polygon(np.column_stack([xs, ys]))


def cobertura_exacta(poly_px) -> Tuple[np.ndarray, int, int]:
    """
    Fracción de cada píxel que el polígono cubre, sin discretizar.

    Args:
        poly_px: Geometría del municipio en coordenadas de píxel continuas.
            Un municipio con islas o con huecos entra aquí igual que uno
            simple: la intersección con la celda ya los tiene en cuenta.

    Returns:
        Tuple con (matriz de pesos en [0,1], fila y columna del origen del
        recorte dentro de la retícula completa). La suma de la matriz es el
        área del municipio en píxeles.
    """
    minx, miny, maxx, maxy = poly_px.bounds
    columna_0, columna_1 = int(np.floor(minx)), int(np.ceil(maxx))
    fila_0, fila_1 = int(np.floor(miny)), int(np.ceil(maxy))

    # La geometría preparada indexa el polígono una vez y responde las consultas
    # de contención en tiempo logarítmico; sin ella esto sería cuadrático.
    preparado = prep(poly_px)
    pesos = np.zeros((fila_1 - fila_0, columna_1 - columna_0))

    for fila in range(fila_0, fila_1):
        for columna in range(columna_0, columna_1):
            celda = box(columna, fila, columna + 1, fila + 1)
            if not preparado.intersects(celda):
                continue
            # Las celdas del interior no necesitan calcular la intersección
            pesos[fila - fila_0, columna - columna_0] = (
                1.0 if preparado.contains(celda) else poly_px.intersection(celda).area
            )

    return pesos, fila_0, columna_0


def metricas_ponderadas(imagen_recortada: np.ndarray, pesos: np.ndarray) -> dict:
    """
    Métricas del municipio ponderadas por el área que cubre de cada píxel.

    Envoltura sobre `core.metricas.metricas_ponderadas` para el caso en que la
    cobertura viene como matriz alineada con el recorte. El cálculo vive en core
    porque `radianza` lo necesita igual, sobre las coberturas precalculadas.

    Args:
        imagen_recortada: Recorte en resolución original
        pesos: Matriz de pesos con la misma forma, en [0,1]

    Returns:
        Diccionario con las métricas, o None si el municipio quedó vacío
    """
    dentro = pesos > 0
    if not dentro.any():
        return None
    return _metricas_nucleo(imagen_recortada[dentro], pesos[dentro])
