import h5py
import math
import numpy as np
import os
from datetime import date
from typing import Any

from ..core.config import IMAGE_PATH, find_image_path
from ..core.lectura import leer_radianza
from ..core.metricas import metricas_ponderadas
from ..core.models import MedicionResultado
from ..core.utils import verificar_georreferencia


def _float_to_json_safe(value: float) -> float | None:
    """Convert float to JSON-serializable value; NaN and Inf become None."""
    if math.isfinite(value):
        return float(value)
    return None


def _normalizar_pesos(entradas) -> list[tuple[int, int, float]]:
    """
    Acepta tripletas (x, y, w) y también el formato anterior (x, y).

    Con el formato anterior no hay más remedio que asumir cobertura 1.0, que es
    justo lo que sesgaba las métricas: cada píxel de frontera cuenta entero
    aunque el municipio solo cubra la mitad. Se avisa en vez de fallar para que
    una tabla sin regenerar no rompa el procesamiento.
    """
    normalizados = []
    incompletas = 0
    for entrada in entradas:
        if len(entrada) == 3:
            x, y, w = entrada
        else:
            (x, y), w = entrada, 1.0
            incompletas += 1
        normalizados.append((int(x), int(y), float(w)))

    if incompletas:
        print(
            f"AVISO: {incompletas} píxeles sin cobertura; se asume 1.0. Las métricas "
            f"quedarán sesgadas por la frontera. Regenera la tabla con "
            f"scripts/generar_coordenadas_pixeles.py"
        )
    return normalizados


def _crop_radiance_and_mask(
    image_matrix: np.ndarray, pesos_validos: list[tuple[int, int, float]]
) -> dict[str, Any]:
    """
    Recorta la matriz de radianza al bounding box del municipio y construye,
    sobre ese recorte, la máscara binaria y la matriz de cobertura.

    La máscara marca los píxeles que el municipio toca; la cobertura dice qué
    fracción de cada uno le corresponde, que es lo que pondera las métricas.
    """
    min_x = min(x for x, _, _ in pesos_validos)
    max_x = max(x for x, _, _ in pesos_validos)
    min_y = min(y for _, y, _ in pesos_validos)
    max_y = max(y for _, y, _ in pesos_validos)

    submatrix = image_matrix[min_y : max_y + 1, min_x : max_x + 1]
    rows, cols = submatrix.shape

    mask = np.zeros((rows, cols), dtype=int)
    coverage = np.zeros((rows, cols), dtype=float)
    for x, y, w in pesos_validos:
        mask[y - min_y, x - min_x] = 1
        coverage[y - min_y, x - min_x] = w

    radiance_list: list[list[float | None]] = [
        [_float_to_json_safe(float(v)) for v in row] for row in submatrix
    ]
    mask_list: list[list[int]] = [list(row) for row in mask]
    coverage_list: list[list[float]] = [[float(v) for v in row] for row in coverage]

    return {
        "bbox": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
        "rows": rows,
        "cols": cols,
        "radiance_matrix": radiance_list,
        "municipality_mask": mask_list,
        "municipality_coverage": coverage_list,
    }


def extract_radiance_matrix(
    downloaded_path: str,
    pesos: list[tuple[int, int, float]],
    date_obj: date,
    municipio: str,
    cuadrante: str | None = None,
) -> dict[str, Any] | None:
    """
    Extrae la submatriz de radianza y la máscara binaria del municipio, recortadas al bounding box.
    Devuelve dict con radiance_matrix, municipality_mask, bbox, rows, cols, municipio, fecha.

    `cuadrante` es opcional solo por compatibilidad: si se pasa, se comprueba
    que el archivo esté georreferenciado donde la tabla de coberturas supone.
    """
    if not os.path.exists(downloaded_path):
        print(f"Archivo no encontrado: {downloaded_path}")
        return None

    try:
        with open(downloaded_path, "rb") as f:
            start = f.read(15)
            if b"<html" in start or b"<!DOCTYPE html" in start:
                print(f"Archivo HTML recibido en vez de HDF5: {downloaded_path}")
                return None

        with h5py.File(downloaded_path, "r") as hdf_file:
            radiance_path = find_image_path(hdf_file)
            image_matrix = hdf_file[radiance_path][()]

            if cuadrante:
                verificar_georreferencia(hdf_file, cuadrante, image_matrix.shape)

            # Filtrar coordenadas válidas dentro de la imagen
            pesos_validos = [
                (x, y, w)
                for x, y, w in _normalizar_pesos(pesos)
                if 0 <= y < image_matrix.shape[0] and 0 <= x < image_matrix.shape[1]
            ]

            if len(pesos_validos) == 0:
                print(
                    f"No se encontraron coordenadas válidas para {municipio} en {date_obj}"
                )
                return None

            crop = _crop_radiance_and_mask(image_matrix, pesos_validos)

            return {
                "municipio": municipio,
                "fecha": date_obj,
                **crop,
            }
    except Exception as e:
        print(f"Error extrayendo matriz de {downloaded_path}: {e}")
        return None


def process_image(downloaded_path, pesos, date_obj, municipio, delete_file=True,
                  cuadrante=None):
    """
    Calcula las métricas del municipio ponderando cada píxel por su cobertura.

    `pesos` son tripletas (x, y, w) con w en (0, 1]. Contar los píxeles enteros
    subestimaba el área entre 7% y 31% según la forma del municipio, porque
    descartaba una franja de frontera que está cubierta a medias.

    Si se pasa `cuadrante`, antes de tocar la radianza se comprueba que el
    archivo declare la misma esquina que se usó para construir las coberturas.
    Sin esa comprobación, un cambio de convención en el producto desplazaría
    todos los municipios en silencio; un solo píxel de desalineamiento mueve la
    media municipal un 4.6% en la mediana de las alcaldías.
    """
    if not os.path.exists(downloaded_path):
        print(f"Archivo no encontrado: {downloaded_path}")
        return None
    
    try:
        with open(downloaded_path, "rb") as f:
            start = f.read(15)
            if b"<html" in start or b"<!DOCTYPE html" in start:
                print(f"Archivo HTML recibido en vez de HDF5: {downloaded_path}")
                return None
        
        with h5py.File(downloaded_path, "r") as hdf_file:
            # La radianza se lee en unidades físicas y con NaN donde no hay
            # medición; `lectura` trae las unidades y la fracción válida.
            image_matrix, lectura = leer_radianza(hdf_file)

            if cuadrante:
                verificar_georreferencia(hdf_file, cuadrante, image_matrix.shape)

            # Filtrar coordenadas válidas
            pesos_validos = [
                (x, y, w) for x, y, w in _normalizar_pesos(pesos)
                if 0 <= y < image_matrix.shape[0] and 0 <= x < image_matrix.shape[1]
            ]

            # Verificar que tenemos coordenadas válidas
            if len(pesos_validos) == 0:
                print(f"⚠️ No se encontraron coordenadas válidas para {municipio} en {date_obj}")
                print(f"   - Total coordenadas: {len(pesos)}")
                print(f"   - Dimensiones imagen: {image_matrix.shape}")
                print(f"   - Rango X: [0, {image_matrix.shape[1]-1}]")
                print(f"   - Rango Y: [0, {image_matrix.shape[0]-1}]")
                return None

            # Informar sobre coordenadas filtradas
            if len(pesos_validos) < len(pesos):
                print(f"ℹ️ Filtradas {len(pesos) - len(pesos_validos)} coordenadas inválidas para {municipio}")
                print(f"   - Coordenadas válidas: {len(pesos_validos)}")
                print(f"   - Coordenadas totales: {len(pesos)}")

            valores = np.array([float(image_matrix[y, x]) for x, y, _ in pesos_validos])
            cobertura = np.array([w for _, _, w in pesos_validos], dtype=float)

            crop = _crop_radiance_and_mask(image_matrix, pesos_validos)
            metricas = metricas_ponderadas(valores, cobertura)
            if metricas is None:
                print(f"⚠️ {municipio} en {date_obj}: ningún píxel con medición válida")
                return None
            if metricas["Fraccion_valida"] < 1.0:
                print(f"⚠️ {municipio} en {date_obj}: solo "
                      f"{metricas['Fraccion_valida']*100:.1f}% del territorio trae medición")

            datos = MedicionResultado(
                Fecha=date_obj,
                Municipio=municipio,
                Unidades_de_radianza=lectura["unidades"] or "nW/(cm2 sr)",
                **metricas,
                Bbox=crop["bbox"],
                Filas=crop["rows"],
                Columnas=crop["cols"],
                Matriz_de_radianza=crop["radiance_matrix"],
                Mascara_municipio=crop["municipality_mask"],
                Cobertura_municipio=crop["municipality_coverage"],
            )
        
        return datos
    
    except Exception as e:
        print(f"Error procesando archivo {downloaded_path}: {e}")
        return None
    
    finally:
        # Eliminar el archivo solo si delete_file=True
        if delete_file:
            try:
                if os.path.exists(downloaded_path):
                    os.remove(downloaded_path)
                    print(f"Archivo eliminado: {downloaded_path}")
            except Exception as e:
                print(f"Error eliminando archivo {downloaded_path}: {e}") 