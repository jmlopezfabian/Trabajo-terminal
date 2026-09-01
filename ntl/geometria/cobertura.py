"""
Cobertura exacta de un municipio sobre la retícula, por intersección geométrica.

`pesos_municipio` aproxima esta misma cantidad subdividiendo cada píxel en k²
subpíxeles y contando los que caen dentro. Es una aproximación con dos cabos
sueltos: el factor k, que hay que elegir, y el peso que se asigna al trazo del
borde, que a k=32 todavía mueve el resultado entre 0.5% y 0.9%.

Aquí no hay ninguno de los dos. El área que el polígono cubre de cada celda se
calcula analíticamente, y cuesta lo mismo que la aproximación porque solo se
paga una vez por municipio. Es también la definición contra la que se validan
`pesos_municipio` y la tabla precalculada.
"""
from typing import Tuple

import numpy as np
from shapely.geometry import Polygon, box
from shapely.prepared import prep

GRADOS_POR_CUADRANTE = 10.0


def poligono_en_pixeles(coordenadas_municipio: np.ndarray,
                        upper_left: Tuple[float, float],
                        shape: Tuple[int, int]) -> Polygon:
    """Convierte el polígono lon/lat a coordenadas de píxel continuas de la retícula."""
    resolucion_x = GRADOS_POR_CUADRANTE / shape[1]
    resolucion_y = GRADOS_POR_CUADRANTE / shape[0]
    xs = (coordenadas_municipio[:, 0] - upper_left[0]) / resolucion_x
    ys = (upper_left[1] - coordenadas_municipio[:, 1]) / resolucion_y
    return Polygon(np.column_stack([xs, ys]))


def cobertura_exacta(poly_px: Polygon) -> Tuple[np.ndarray, int, int]:
    """
    Fracción de cada píxel que el polígono cubre, sin discretizar.

    Args:
        poly_px: Polígono del municipio en coordenadas de píxel continuas

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
