import os
import tempfile
from dotenv import load_dotenv
from importlib import resources
from pathlib import Path

load_dotenv()


def _primera_definida(*nombres: str) -> str | None:
    """
    Primer valor no vacío de una lista de variables de entorno.

    Las variables llevaban el prefijo VNP46A1_, que era el nombre anterior del
    paquete. Se leen los dos nombres para no romper un despliegue que ya tenga
    el antiguo configurado; el nuevo tiene precedencia.
    """
    for nombre in nombres:
        valor = os.getenv(nombre)
        if valor:
            return valor
    return None


BASE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A1/{year}/{day}/"
IMAGE_PATH = "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m"

# Where the downloaded HDF5 files (hundreds of MB each) are staged.
#
# This used to be the literal relative path "../temp", which resolved against
# whatever the current working directory happened to be — so the same code
# wrote to a different place depending on how the process was started, and in a
# container could fill the root filesystem instead of the intended volume.
# It is now an absolute path: set NTL_TEMP_DIR to control it, otherwise it
# falls back to a subdirectory of the system temp dir, which is always writable
# and never depends on the cwd.
_DEFAULT_TEMP_DIR = Path(tempfile.gettempdir()) / "ntl"
TEMP_DIR = Path(
    _primera_definida("NTL_TEMP_DIR", "VNP46A1_TEMP_DIR") or _DEFAULT_TEMP_DIR
).expanduser().resolve()


def temp_path(filename: str) -> Path:
    """Absolute path for a staged file, creating TEMP_DIR on first use."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR / filename


# Where the Parquet results are written.
#
# Esto era "../data", que además de depender del cwd escribía en el directorio
# padre: lanzar el proceso desde una subcarpeta dejaba los resultados fuera del
# proyecto. A diferencia de TEMP_DIR, el valor por omisión no es el temporal del
# sistema: son resultados, no archivos de paso, y perderlos ahí sería peor.
# Usa NTL_DATA_DIR para fijarlo.
DATA_DIR = Path(
    _primera_definida("NTL_DATA_DIR", "VNP46A1_DATA_DIR") or Path.cwd() / "data"
).expanduser().resolve()


def data_path(filename: str) -> Path:
    """Absolute path for a result file, creating DATA_DIR on first use."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / filename


def find_image_path(hdf_file) -> str:
    """
    Encuentra la ruta al dataset de radianza. Colección 5200 puede usar
    VIIRS_Grid_DNB_2d o VNP_Grid_DNB; busca alternativas si el path estándar falla.
    """
    candidates = [
        IMAGE_PATH,
        "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_At_Sensor_Radiance_500m",
        "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_At_Sensor_Radiance",
    ]
    for path in candidates:
        try:
            if path in hdf_file:
                return path
        except Exception:
            pass

    try:
        if "HDFEOS" in hdf_file and "GRIDS" in hdf_file["HDFEOS"]:
            grids = hdf_file["HDFEOS"]["GRIDS"]
            for grid_name in list(grids.keys()):
                grid = grids[grid_name]
                for key in list(grid.keys()):
                    if "Data" in key and "Field" in key:
                        data_fields = grid[key]
                        for field_name in list(data_fields.keys()):
                            if "Radiance" in field_name and "DNB" in field_name:
                                path = f"HDFEOS/GRIDS/{grid_name}/{key}/{field_name}"
                                try:
                                    if path in hdf_file:
                                        return path
                                except Exception:
                                    pass
    except Exception:
        pass

    return IMAGE_PATH
_DATA_ROOT = resources.files("ntl_data")

# Polígonos municipales: entrada de ntl.geometria, que los convierte en
# coberturas por píxel.
RUTA_MUNICIPIOS = str(_DATA_ROOT.joinpath("limite-de-las-alcaldias.json"))

# Coberturas ya calculadas: entrada de ntl.radianza, que las aplica a cada
# imagen diaria sin recalcular geometría.
PIXELES_MUNICIPIOS = str(_DATA_ROOT.joinpath("municipios_coordenadas_pixeles.json"))

TOKEN = os.getenv("NASA_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

# Tamaño de bloque para la descarga bloqueante
CHUNK_SIZE = 8192