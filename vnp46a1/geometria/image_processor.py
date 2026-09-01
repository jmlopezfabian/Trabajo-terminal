import numpy as np
from scipy import ndimage
from typing import Tuple, List
from ..core.utils import distancia_puntos

def aumentar_imagen(image_matrix: np.ndarray, factor_escala: int) -> np.ndarray:
    """Aumenta el tamaño de una imagen por un factor de escala"""
    imagen_aumentada = np.kron(image_matrix, np.ones((factor_escala, factor_escala)))
    return imagen_aumentada

def recortar(image_matrix: np.ndarray, coordenadas_municipio: np.ndarray,
             upper_left: Tuple[float, float], factor_escala: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recorta la imagen al bounding box del municipio SIN aumentarla.

    El recorte se queda en la resolución original: el factor de escala solo se
    aplica a las coordenadas del borde, que es lo único que necesita la malla
    fina. Los valores de radianza son constantes dentro de cada píxel original,
    así que replicarlos no aporta información y sí multiplica la memoria por k².

    Returns:
        Tuple con (recorte en resolución original, x del borde en malla fina,
        y del borde en malla fina)
    """
    resolucion_x = 10 / image_matrix.shape[1]
    resolucion_y = 10 / image_matrix.shape[0]

    x_pixels = (coordenadas_municipio[:, 0] - upper_left[0]) / resolucion_x
    y_pixels = (upper_left[1] - coordenadas_municipio[:, 1]) / resolucion_y

    recorte_y = (np.ceil(y_pixels.min()).astype(int)-1, np.ceil(y_pixels.max()).astype(int)+1)
    recorte_x = (np.ceil(x_pixels.min()).astype(int)-1, np.ceil(x_pixels.max()).astype(int)+1)

    image_matrix_recortada = image_matrix[recorte_y[0]:recorte_y[1], recorte_x[0]:recorte_x[1]]

    nuevos_x_pixels = (x_pixels - recorte_x[0]) * factor_escala
    nuevos_y_pixels = (y_pixels - recorte_y[0]) * factor_escala
    return image_matrix_recortada, nuevos_x_pixels, nuevos_y_pixels


def recortar_imagen(image_matrix: np.ndarray, coordenadas_municipio: np.ndarray,
                   upper_left: Tuple[float, float], factor_escala: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recorta una imagen según las coordenadas del municipio y aplica factor de escala.

    Materializa el producto Kronecker sobre la radianza. Se conserva para
    compatibilidad y para inspección visual; el flujo de medición usa `recortar`
    más `pesos_municipio`, que no necesita replicar los valores.

    Args:
        image_matrix: Matriz de la imagen original
        coordenadas_municipio: Coordenadas del municipio
        upper_left: Coordenadas de la esquina superior izquierda
        factor_escala: Factor de escala para aumentar la imagen

    Returns:
        Tuple con (imagen_aumentada, nuevos_x_pixels, nuevos_y_pixels)
    """
    recortada, nuevos_x_pixels, nuevos_y_pixels = recortar(
        image_matrix, coordenadas_municipio, upper_left, factor_escala
    )

    if factor_escala == 1:
        imagen_aumentada = recortada
    else:
        imagen_aumentada = aumentar_imagen(recortada, factor_escala)

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

def pesos_municipio(shape_recorte: Tuple[int, int], nuevos_x_pixels: np.ndarray,
                    nuevos_y_pixels: np.ndarray, factor_escala: int = 1,
                    peso_borde: float = 0.5) -> np.ndarray:
    """
    Calcula qué fracción de cada píxel ORIGINAL pertenece al municipio.

    Aquí es donde entra el producto Kronecker, y solo aquí: la malla fina se
    construye para decidir geometría, sobre una máscara booleana, y se colapsa
    de inmediato a un peso por píxel original. La radianza nunca se replica.

        malla fina (h·k × w·k, bool)  ->  pesos (h × w, float en [0,1])

    Args:
        shape_recorte: Forma (alto, ancho) del recorte en resolución original
        nuevos_x_pixels: X de los vértices ya escalados por factor_escala
        nuevos_y_pixels: Y de los vértices ya escalados por factor_escala
        factor_escala: Subdivisiones por lado de cada píxel original
        peso_borde: Cuánto cuenta un subpíxel por el que pasa el trazo del
            borde. El trazo cae a caballo sobre la frontera, así que 0.5 es el
            valor insesgado; 0.0 reproduce el comportamiento histórico
            (descartar la frontera) y 1.0 la incluye entera.

    Returns:
        Matriz de pesos con la forma del recorte original. Su suma es el área
        del municipio en píxeles originales.
    """
    alto, ancho = shape_recorte
    k = factor_escala
    alto_fino, ancho_fino = alto * k, ancho * k

    bordes = completar_bordes(nuevos_x_pixels, nuevos_y_pixels)

    mascara_borde = np.zeros((alto_fino, ancho_fino), dtype=bool)
    if bordes:
        bx = np.fromiter((p[0] for p in bordes), dtype=np.intp, count=len(bordes))
        by = np.fromiter((p[1] for p in bordes), dtype=np.intp, count=len(bordes))
        dentro = (bx >= 0) & (bx < ancho_fino) & (by >= 0) & (by < alto_fino)
        mascara_borde[by[dentro], bx[dentro]] = True

    # Todo lo que el exterior no alcanza y no es borde es interior, sin importar
    # en cuántas componentes conexas esté partido el municipio.
    #
    # `out=` explícito en vez de `~mascara_borde`: la máscara se vuelve a usar
    # más abajo, y numpy 1.x sobre Python 3.14 elide mal los temporales y la
    # sobreescribía en sitio a partir de 256 KB. requirements.txt ya exige
    # numpy>=2.1, donde no ocurre; esto lo deja explícito y a prueba de entornos.
    libre = np.logical_not(mascara_borde, out=np.empty_like(mascara_borde))
    etiquetas, _ = ndimage.label(libre)
    marco = np.concatenate([
        etiquetas[0, :], etiquetas[-1, :], etiquetas[:, 0], etiquetas[:, -1]
    ])
    etiquetas_exteriores = np.unique(marco)
    etiquetas_exteriores = etiquetas_exteriores[etiquetas_exteriores != 0]
    interior = libre & ~np.isin(etiquetas, etiquetas_exteriores)

    fino = interior.astype(np.float64)
    if peso_borde:
        fino += peso_borde * mascara_borde

    # Colapsar la malla fina: promedio de subpíxeles dentro de cada píxel original
    return fino.reshape(alto, k, ancho, k).mean(axis=(1, 3))


def metricas_ponderadas(imagen_recortada: np.ndarray, pesos: np.ndarray) -> dict:
    """
    Métricas del municipio ponderadas por el área que cubre de cada píxel.

    Son invariantes al factor de escala: subir k refina los pesos, no multiplica
    las cantidades. `Cantidad_de_pixeles` es un área en píxeles originales, no un
    conteo de subpíxeles, así que es comparable entre corridas con distinta k.

    Args:
        imagen_recortada: Recorte en resolución original
        pesos: Matriz de pesos con la misma forma, en [0,1]

    Returns:
        Diccionario con las métricas, o None si el municipio quedó vacío
    """
    dentro = pesos > 0
    if not dentro.any():
        return None

    valores = imagen_recortada[dentro].astype(np.float64)
    w = pesos[dentro]

    area = float(w.sum())
    suma = float(np.dot(w, valores))
    media = suma / area
    varianza = float(np.dot(w, (valores - media) ** 2) / area)

    # Percentiles ponderados: posición de cada valor en la masa acumulada de área
    orden = np.argsort(valores)
    valores_ord, w_ord = valores[orden], w[orden]
    acumulada = (np.cumsum(w_ord) - 0.5 * w_ord) / area
    p25, p50, p75 = np.interp([0.25, 0.50, 0.75], acumulada, valores_ord)

    return {
        "Cantidad_de_pixeles": area,
        "Suma_de_radianza": suma,
        "Media_de_radianza": media,
        "Desviacion_estandar_de_radianza": float(np.sqrt(varianza)),
        "Maximo_de_radianza": float(valores.max()),
        "Minimo_de_radianza": float(valores.min()),
        "Percentil_25_de_radianza": float(p25),
        "Percentil_50_de_radianza": float(p50),
        "Percentil_75_de_radianza": float(p75),
    }


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
