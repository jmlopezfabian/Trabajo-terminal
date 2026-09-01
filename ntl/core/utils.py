import json
import re
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple

from .config import RUTA_MUNICIPIOS
from .models import CoordenadasPixeles


def normalize_municipio(municipio: str) -> str:
    """Normaliza el nombre del municipio removiendo acentos"""
    return municipio.lower().replace("á", "a").replace("é", "e")\
        .replace("í", "i").replace("ó", "o").replace("ú", "u")


def parse_date(date_str: str) -> Tuple[int, str, datetime.date]:
    """
    Parsea una fecha en formato dd-mm-yy y retorna año, día del año y objeto date.

    El día se devuelve con tres dígitos porque así lo publica el archivo de
    LAADS; las dos versiones del procesamiento diferían en esto y solo una
    construía la URL en el formato canónico.
    """
    date = datetime.strptime(date_str, "%d-%m-%y")
    day_of_year = date.timetuple().tm_yday
    return date.year, f"{day_of_year:03d}", date.date()


def load_coord_data(municipio: str, path: str) -> CoordenadasPixeles:
    """Carga las coordenadas de píxeles de un municipio desde un archivo JSON"""
    with open(path, "r") as f:
        data = json.load(f)
    return CoordenadasPixeles(**data[municipio])


def extraer_coordenadas(nombre_delegacion: str) -> Optional[np.ndarray]:
    """Extrae las coordenadas del polígono de un municipio desde el archivo de límites"""
    try:
        with open(RUTA_MUNICIPIOS, "r") as file:
            datos = json.load(file)

        for data in datos["features"]:
            if data["properties"]["NOMGEO"] == nombre_delegacion:
                return np.array(data["geometry"]["coordinates"][0])

        print(f"No se encontraron coordenadas para la delegación: {nombre_delegacion}")
        return None
    except Exception as e:
        print(f"Error al extraer coordenadas: {e}")
        return None


def left_right_coords(hdf_file) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Extrae las coordenadas de la esquina superior izquierda e inferior derecha del archivo HDF"""
    try:
        dataset_path = "HDFEOS INFORMATION/StructMetadata.0"
        metadata = hdf_file[dataset_path][()].tobytes().decode("utf-8")

        upper_left_match = re.search(r"UpperLeftPointMtrs=\(([-\d.]+),([-\d.]+)\)", metadata)
        lower_right_match = re.search(r"LowerRightMtrs=\(([-\d.]+),([-\d.]+)\)", metadata)
        conversion = 1_000_000

        if upper_left_match and lower_right_match:
            upper_left_coords = (
                float(upper_left_match.group(1)) / conversion,
                float(upper_left_match.group(2)) / conversion,
            )
            lower_right_coords = (
                float(lower_right_match.group(1)) / conversion,
                float(lower_right_match.group(2)) / conversion,
            )
            return upper_left_coords, lower_right_coords

        print("No se pudieron extraer las coordenadas.")
        return None, None
    except Exception as e:
        print(f"Error al extraer coordenadas del archivo HDF: {e}")
        return None, None


_CUADRANTE_RE = re.compile(r"^h(\d{2})v(\d{2})$")

# Lado de un cuadrante del grid de Black Marble, en grados.
GRADOS_POR_CUADRANTE = 10.0


def esquina_superior_izquierda(cuadrante: str) -> Tuple[float, float]:
    """
    Longitud y latitud de la esquina superior izquierda de un cuadrante hHHvVV.

    VNP46A1 se publica sobre una retícula lat/lon lineal (``Projection=
    HE5_GCTP_GEO``), no sinusoidal como MODIS, así que el origen de cada
    cuadrante se deriva de su identificador sin necesidad de abrir el archivo.
    El grid está anclado a la esquina, no al centro del píxel
    (``GridOrigin=HE5_HDFE_GD_UL``): el píxel (0, 0) cubre
    ``[lon, lon+res) x (lat-res, lat]``.

    Esta es la única definición del origen en el proyecto; estaba duplicada en
    el script generador y en las pruebas, que es exactamente la clase de
    constante que no conviene tener por triplicado.
    """
    coincidencia = _CUADRANTE_RE.match(cuadrante)
    if not coincidencia:
        raise ValueError(f"Cuadrante mal formado: {cuadrante!r}; se esperaba hHHvVV")
    h, v = int(coincidencia.group(1)), int(coincidencia.group(2))
    return (-180.0 + GRADOS_POR_CUADRANTE * h, 90.0 - GRADOS_POR_CUADRANTE * v)


def cuadrante_de_coordenadas(coordenadas: np.ndarray) -> str:
    """
    Cuadrante que contiene un polígono, derivado de sus propias coordenadas.

    Sirve para no heredar el identificador de una tabla previa. Falla si el
    polígono cruza de cuadrante: ese caso necesita leer dos imágenes y componer
    el resultado, y devolver uno de los dos en silencio recortaría el municipio.
    """
    lon_min, lon_max = float(coordenadas[:, 0].min()), float(coordenadas[:, 0].max())
    lat_min, lat_max = float(coordenadas[:, 1].min()), float(coordenadas[:, 1].max())

    h_min = int((lon_min + 180.0) // GRADOS_POR_CUADRANTE)
    h_max = int((lon_max + 180.0) // GRADOS_POR_CUADRANTE)
    v_min = int((90.0 - lat_max) // GRADOS_POR_CUADRANTE)
    v_max = int((90.0 - lat_min) // GRADOS_POR_CUADRANTE)

    if h_min != h_max or v_min != v_max:
        raise ValueError(
            f"El polígono cruza de cuadrante: h{h_min:02d}-h{h_max:02d} "
            f"v{v_min:02d}-v{v_max:02d}. Componer varios cuadrantes no está "
            f"implementado."
        )
    return f"h{h_min:02d}v{v_min:02d}"


def verificar_georreferencia(hdf_file, cuadrante: str, forma: Tuple[int, int] | None = None,
                             tolerancia_px: float = 0.01) -> Tuple[float, float]:
    """
    Comprueba que la imagen está donde el código supone que está.

    Toda la tabla de coberturas se construye sobre un origen *derivado del
    identificador del cuadrante*, sin mirar el archivo. Es un supuesto, y hasta
    ahora nada lo confrontaba con la realidad: `left_right_coords` existía pero
    nadie la llamaba. Medio píxel de error desplazaría cada municipio ~230 m de
    forma sistemática, y las pruebas de área no lo verían, porque validan el
    rasterizado contra el mismo supuesto.

    Ese desplazamiento no es inocuo: mover la retícula un solo píxel cambia la
    media de radianza municipal un 4.6% (mediana sobre las 16 alcaldías) y hasta
    un 12.2% en Xochimilco. Ver ``scripts/sensibilidad_desplazamiento.py``.

    Args:
        hdf_file: Archivo HDF5 abierto
        cuadrante: Identificador hHHvVV con el que se calcularon las coberturas
        forma: Forma (alto, ancho) de la matriz de radianza, para expresar la
            discrepancia en píxeles. Si se omite, se asume 2400x2400.
        tolerancia_px: Discrepancia máxima aceptable, en píxeles

    Returns:
        La esquina superior izquierda declarada por el archivo

    Raises:
        ValueError: Si el archivo dice estar en otro sitio del que se supone
    """
    ul_archivo, lr_archivo = left_right_coords(hdf_file)
    if ul_archivo is None or lr_archivo is None:
        # Sin metadatos no hay nada que comparar. No es motivo para tirar el
        # procesamiento: se avisa y se sigue con el supuesto de siempre.
        print(f"AVISO: {cuadrante} no trae StructMetadata.0 legible; no se pudo "
              f"verificar la georreferencia.")
        return esquina_superior_izquierda(cuadrante)

    alto, ancho = forma if forma else (2400, 2400)
    resolucion_x = GRADOS_POR_CUADRANTE / ancho
    resolucion_y = GRADOS_POR_CUADRANTE / alto

    esperada = esquina_superior_izquierda(cuadrante)
    error_x = abs(ul_archivo[0] - esperada[0]) / resolucion_x
    error_y = abs(ul_archivo[1] - esperada[1]) / resolucion_y

    if error_x > tolerancia_px or error_y > tolerancia_px:
        raise ValueError(
            f"Georreferencia incoherente para {cuadrante}: el archivo declara "
            f"su esquina en {ul_archivo}, el código la supone en {esperada} "
            f"({error_x:.3f} px en x, {error_y:.3f} px en y). Las coberturas "
            f"precalculadas apuntarían al lugar equivocado."
        )

    # El cuadrante debe abarcar 10 grados; si no, la resolución supuesta al
    # construir la tabla no es la del archivo y el error crece con la distancia
    # al origen aunque la esquina coincida.
    ancho_grados = abs(lr_archivo[0] - ul_archivo[0])
    alto_grados = abs(ul_archivo[1] - lr_archivo[1])
    if (abs(ancho_grados - GRADOS_POR_CUADRANTE) > tolerancia_px * resolucion_x
            or abs(alto_grados - GRADOS_POR_CUADRANTE) > tolerancia_px * resolucion_y):
        raise ValueError(
            f"Extensión inesperada para {cuadrante}: {ancho_grados}x{alto_grados} "
            f"grados en vez de {GRADOS_POR_CUADRANTE}x{GRADOS_POR_CUADRANTE}."
        )

    return ul_archivo


def distancia_puntos(x: Tuple[float, float], y: Tuple[float, float]) -> float:
    """Calcula la distancia euclidiana entre dos puntos"""
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)


def polygon_centroid(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calcula el centroide de un polígono"""
    n = len(vertices)
    A = 0
    Cx = 0
    Cy = 0

    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross

    A *= 0.5
    Cx /= (6 * A)
    Cy /= (6 * A)
    return (Cx, Cy)
