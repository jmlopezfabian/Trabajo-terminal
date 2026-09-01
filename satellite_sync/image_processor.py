import numpy as np
from scipy import ndimage
from typing import Tuple, List
from .utils import distancia_puntos

def aumentar_imagen(image_matrix: np.ndarray, factor_escala: int) -> np.ndarray:
    """Aumenta el tamaño de una imagen por un factor de escala"""
    imagen_aumentada = np.kron(image_matrix, np.ones((factor_escala, factor_escala)))
    return imagen_aumentada

def recortar_imagen(image_matrix: np.ndarray, coordenadas_municipio: np.ndarray, 
                   upper_left: Tuple[float, float], factor_escala: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recorta una imagen según las coordenadas del municipio y aplica factor de escala.
    
    Args:
        image_matrix: Matriz de la imagen original
        coordenadas_municipio: Coordenadas del municipio
        upper_left: Coordenadas de la esquina superior izquierda
        factor_escala: Factor de escala para aumentar la imagen
        
    Returns:
        Tuple con (imagen_aumentada, nuevos_x_pixels, nuevos_y_pixels)
    """
    # 1. Calcular resolución en la imagen original
    resolucion_x = 10 / image_matrix.shape[1]
    resolucion_y = 10 / image_matrix.shape[0]

    # 2. Convertir coordenadas a píxeles
    x_pixels = (coordenadas_municipio[:, 0] - upper_left[0]) / resolucion_x
    y_pixels = (upper_left[1] - coordenadas_municipio[:, 1]) / resolucion_y

    # 3. Definir área de recorte
    recorte_y = (np.ceil(y_pixels.min()).astype(int)-1, np.ceil(y_pixels.max()).astype(int)+1)
    recorte_x = (np.ceil(x_pixels.min()).astype(int)-1, np.ceil(x_pixels.max()).astype(int)+1)

    # 4. Recortar la imagen original
    image_matrix_recortada = image_matrix[recorte_y[0]:recorte_y[1], recorte_x[0]:recorte_x[1]]

    # 5. Aumentar imagen recortada
    if factor_escala == 1:
        imagen_aumentada = image_matrix_recortada
    else:
        imagen_aumentada = aumentar_imagen(image_matrix_recortada, factor_escala)

    # 6. Ajustar las coordenadas de los bordes según el factor de escala
    nuevos_x_pixels = (x_pixels - recorte_x[0]) * factor_escala
    nuevos_y_pixels = (y_pixels - recorte_y[0]) * factor_escala
    return imagen_aumentada, nuevos_x_pixels, nuevos_y_pixels

def completar_bordes(nuevos_x_pixels: np.ndarray, nuevos_y_pixels: np.ndarray) -> List[Tuple[int, int]]:
    """
    Completa los bordes del polígono interpolando puntos entre vértices distantes.

    La densidad de muestreo se deriva de la longitud de cada arista, no de una
    constante: una arista de d píxeles se muestrea con 2d+1 puntos, de modo que
    nunca quedan huecos por los que el relleno pueda fugarse, sea cual sea el
    factor de escala. La interpolación es paramétrica en t en vez de despejar
    y en función de x, así que las aristas verticales (dx = 0) no producen
    pendientes infinitas.

    Args:
        nuevos_x_pixels: Coordenadas X de los vértices, en la malla de destino
        nuevos_y_pixels: Coordenadas Y de los vértices, en la malla de destino

    Returns:
        Lista de coordenadas enteras que forman un borde cerrado sin huecos
    """
    coordenadas_bordes = []
    vistos = set()  # membresía en O(1); con una lista el trazado es cuadrático

    def agregar(x: float, y: float) -> None:
        punto = (int(x), int(y))
        if punto not in vistos:
            vistos.add(punto)
            coordenadas_bordes.append(punto)

    for i in range(len(nuevos_x_pixels) - 1):
        x0, y0 = nuevos_x_pixels[i], nuevos_y_pixels[i]
        x1, y1 = nuevos_x_pixels[i+1], nuevos_y_pixels[i+1]
        agregar(x0, y0)

        distancia = distancia_puntos((x0, y0), (x1, y1))
        if distancia > 1:
            # Dos muestras por píxel de arista: suficiente para que dos puntos
            # consecutivos caigan en celdas iguales o adyacentes.
            n_puntos = int(np.ceil(distancia)) * 2 + 1
            t = np.linspace(0.0, 1.0, n_puntos)
            xs = x0 + (x1 - x0) * t
            ys = y0 + (y1 - y0) * t
            for x, y in zip(xs, ys):
                agregar(x, y)

    agregar(nuevos_x_pixels[-1], nuevos_y_pixels[-1])

    return coordenadas_bordes

def get_pixeles(imagen: np.ndarray, bordes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Obtiene todos los píxeles interiores del municipio en un solo paso.

    En lugar de sembrar un flood fill en el centroide (que puede caer fuera del
    polígono si este es cóncavo, y que deja "huérfanas" las regiones interiores
    no conectadas con la semilla), se inunda desde el marco de la imagen hacia
    adentro. Todo lo que el exterior no alcanza y no es borde es interior, sin
    importar en cuántas componentes conexas esté partido.

    Args:
        imagen: Matriz de la imagen recortada
        bordes: Lista de coordenadas (x, y) que forman el borde del polígono

    Returns:
        Lista de coordenadas (x, y) de los píxeles dentro del polígono
    """
    height, width = imagen.shape

    # Máscara de borde: O(1) por consulta, en vez de buscar en una lista
    mascara_borde = np.zeros((height, width), dtype=bool)
    for x, y in bordes:
        if 0 <= x < width and 0 <= y < height:
            mascara_borde[y, x] = True

    # Componentes conexas (4-vecinos) de todo lo que no es borde
    libre = ~mascara_borde
    etiquetas, _ = ndimage.label(libre)

    # Las componentes que tocan el marco de la imagen son el exterior
    marco = np.concatenate([
        etiquetas[0, :], etiquetas[-1, :], etiquetas[:, 0], etiquetas[:, -1]
    ])
    etiquetas_exteriores = np.unique(marco)
    etiquetas_exteriores = etiquetas_exteriores[etiquetas_exteriores != 0]

    interior = libre & ~np.isin(etiquetas, etiquetas_exteriores)

    ys, xs = np.nonzero(interior)
    return list(zip(xs.tolist(), ys.tolist()))
