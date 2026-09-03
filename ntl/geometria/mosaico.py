"""
Municipios que no caben en un cuadrante.

La retícula de Black Marble se publica cortada en cuadrantes de 10°x10° por
conveniencia del archivo, no por ninguna frontera administrativa. Un municipio
pegado a un múltiplo de 10 en longitud cae en dos cuadrantes, en latitud cae en
dos, y cerca de una esquina de la retícula cae en cuatro. Hasta ahora ese caso
se detectaba y se rechazaba: el municipio se omitía de la tabla de cobertura.

La idea de este módulo es no razonar nunca sobre "el cuadrante de un municipio".
Existe **una sola retícula global**, anclada en (-180, 90), y los cuadrantes son
un troceado de ella. La cobertura se calcula una vez sobre esa retícula global
—donde el municipio es una figura continua sin costuras— y solo después se
reparte en piezas, una por cuadrante.

Que el reparto sea exacto no es casualidad ni aproximación: el lado del
cuadrante son 2400 píxeles enteros, así que el borde entre cuadrantes cae
siempre en un borde de píxel. **Ningún píxel se parte entre dos cuadrantes.** La
suma de las áreas de las piezas es idénticamente el área global, y esa igualdad
es lo que se comprueba en las pruebas.
"""
from typing import Dict, List, Tuple

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .cobertura import GRADOS_POR_CUADRANTE, cobertura_exacta
from ..core.utils import cuadrantes_de_coordenadas

# Los cuadrantes se nombran hHHvVV; se descompone aquí para no repetir el regex.
def indices_cuadrante(cuadrante: str) -> Tuple[int, int]:
    """Índices (h, v) de un cuadrante hHHvVV."""
    return int(cuadrante[1:3]), int(cuadrante[4:6])


def poligono_en_pixeles_globales(coordenadas_municipio,
                                 forma_cuadrante: Tuple[int, int]):
    """
    Límite del municipio en coordenadas de píxel de la retícula global.

    El origen es (-180, 90), la esquina superior izquierda de h00v00, y la
    resolución es la del cuadrante. La diferencia con
    `cobertura.poligono_en_pixeles` es solo el origen: allí el marco es el de un
    cuadrante concreto, aquí el del planeta, que es el único marco en el que un
    municipio a caballo entre dos imágenes sigue siendo una figura continua.

    Acepta una geometría de shapely —con islas y huecos— o un arreglo de vértices.
    """
    alto, ancho = forma_cuadrante
    resolucion_x = GRADOS_POR_CUADRANTE / ancho
    resolucion_y = GRADOS_POR_CUADRANTE / alto

    if isinstance(coordenadas_municipio, BaseGeometry):
        return affine_transform(coordenadas_municipio, [
            1 / resolucion_x, 0.0,
            0.0, -1 / resolucion_y,
            180.0 / resolucion_x, 90.0 / resolucion_y,
        ])

    coordenadas_municipio = np.asarray(coordenadas_municipio)
    xs = (coordenadas_municipio[:, 0] + 180.0) / resolucion_x
    ys = (90.0 - coordenadas_municipio[:, 1]) / resolucion_y
    return Polygon(np.column_stack([xs, ys]))


def partes(geometria):
    """
    Las piezas conexas de una geometría, siempre como lista.

    Un municipio con islas se cubre parte por parte y no de una sola pasada: el
    recorrido de `cobertura_exacta` va celda a celda por la envolvente, y la
    envolvente de un municipio con una isla a 80 km incluye los 80 km de mar que
    hay en medio. Recorrerlos costaría minutos para no encontrar un solo píxel.
    """
    return list(geometria.geoms) if hasattr(geometria, "geoms") else [geometria]


def cobertura_por_cuadrante(coordenadas_municipio: np.ndarray,
                            forma_cuadrante: Tuple[int, int] = (2400, 2400),
                            ) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    Reparte la cobertura de un municipio entre los cuadrantes que toca.

    Args:
        coordenadas_municipio: Límite del municipio, como geometría de shapely
            —con sus islas y sus huecos— o como arreglo de vértices lon/lat
        forma_cuadrante: (alto, ancho) en píxeles de un cuadrante del producto

    Returns:
        Diccionario {cuadrante: [(x, y, w), ...]} con x, y **locales** a la
        retícula de ese cuadrante y w en (0, 1]. Los cuadrantes que la
        envolvente toca pero la geometría no —el caso de un municipio en L,
        cuya caja abarca cuatro y su territorio tres— no aparecen: descargarlos
        costaría cientos de megas para no aportar un solo píxel.

    Raises:
        ValueError: Si el polígono cruza el antimeridiano.
    """
    alto, ancho = forma_cuadrante

    # El cálculo geométrico se hace en el marco global, donde el municipio no
    # tiene costuras; repartir después es aritmética entera. Calcular la
    # cobertura cuadrante por cuadrante habría significado recortar el polígono
    # contra cada borde y confiar en que los trozos vuelven a sumar el original.
    #
    # Se recorre parte por parte y no la geometría entera: la envolvente de un
    # municipio con una isla a 80 km incluye el mar que hay en medio, y
    # `cobertura_exacta` visita cada celda de la envolvente que se le da.
    global_px = poligono_en_pixeles_globales(coordenadas_municipio, forma_cuadrante)

    acumulado: Dict[Tuple[int, int], float] = {}
    for parte in partes(global_px):
        if parte.is_empty:
            continue
        pesos, fila_0, columna_0 = cobertura_exacta(parte)
        filas, columnas = np.nonzero(pesos)
        for fila, columna in zip(filas, columnas):
            clave = (int(columna) + columna_0, int(fila) + fila_0)
            # Las partes de un municipio son disjuntas, pero dos que se tocan
            # comparten los píxeles de su frontera común: ahí las áreas se
            # suman, que es lo que haría el municipio entero.
            acumulado[clave] = acumulado.get(clave, 0.0) + float(pesos[fila, columna])

    # Se valida la envolvente aunque el reparto no la necesite: es donde se
    # rechaza el antimeridiano, y sirve de cota superior con la que contrastar.
    esperados = set(cuadrantes_de_coordenadas(coordenadas_municipio))

    piezas: Dict[str, List[Tuple[int, int, float]]] = {}
    for (x_global, y_global), peso in acumulado.items():
        # Una esquina puede rozar una celda y dejar 1e-8 de cobertura; al
        # redondear queda en cero y no aporta ni área ni valor.
        w = round(min(peso, 1.0), 6)
        if w <= 0:
            continue
        h, x = divmod(x_global, ancho)
        v, y = divmod(y_global, alto)
        piezas.setdefault(f"h{h:02d}v{v:02d}", []).append((x, y, w))

    sobrantes = set(piezas) - esperados
    if sobrantes:
        raise ValueError(
            f"El reparto asignó píxeles a cuadrantes fuera de la envolvente: "
            f"{sorted(sobrantes)}. La retícula global y la envolvente no "
            f"coinciden, que es la clase de error que desplaza un municipio entero."
        )

    # De arriba a abajo y de izquierda a derecha, el mismo orden de lectura
    # que devuelve `cuadrantes_de_coordenadas`; ordenar por el nombre pondría
    # h08v08 antes que h09v07, que no es como se recorre un mosaico.
    orden = sorted(piezas, key=lambda c: tuple(reversed(indices_cuadrante(c))))
    # Orden de lectura también dentro de la pieza: el recorrido por partes
    # las produce agrupadas por isla, y la tabla no debe depender de eso.
    return {c: sorted(piezas[c], key=lambda t: (t[1], t[0])) for c in orden}


def cuadrante_referencia(cuadrantes: List[str]) -> str:
    """
    Cuadrante cuyo origen sirve de marco para el recorte y el bounding box.

    Es el del extremo noroeste (menor h, menor v) de los que el municipio toca.
    Con un solo cuadrante es ese mismo, así que el bbox de un municipio que cabe
    en una imagen no cambia de significado respecto a lo que ya se publicaba.
    """
    if not cuadrantes:
        raise ValueError("No hay cuadrantes de los que tomar referencia")
    h = min(indices_cuadrante(c)[0] for c in cuadrantes)
    v = min(indices_cuadrante(c)[1] for c in cuadrantes)
    return f"h{h:02d}v{v:02d}"


def a_marco_referencia(cuadrante: str, referencia: str,
                       forma_cuadrante: Tuple[int, int] = (2400, 2400),
                       ) -> Tuple[int, int]:
    """
    Desplazamiento (columna, fila) del origen de un cuadrante en el marco de la
    referencia. Con `cuadrante == referencia` es (0, 0).
    """
    alto, ancho = forma_cuadrante
    h, v = indices_cuadrante(cuadrante)
    h_ref, v_ref = indices_cuadrante(referencia)
    return (h - h_ref) * ancho, (v - v_ref) * alto
