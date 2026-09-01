"""
Lectura de la radianza desde el HDF5, en unidades físicas y sin valores de relleno.

El dataset viene como enteros sin signo con tres atributos que hay que respetar:
`scale_factor` y `add_offset` para llegar a nW/(cm² sr), y `_FillValue` junto con
`valid_min`/`valid_max` para saber qué píxeles no traen medición.

Ignorar el relleno no es un detalle: en el histórico hay cuatro fechas en las que
el cuadrante completo venía relleno, y como 65535 se sumó como si fuera radianza,
esos registros salieron con sumas 229 veces mayores que las normales.
"""
from typing import Tuple

import numpy as np

from .config import BANDERA_CALIDAD, CALIDAD_ACEPTABLE, find_image_path


def _buscar_hermano(hdf_file, dataset, nombre):
    """Otro dataset del mismo grupo `Data Fields`, si existe."""
    grupo = dataset.name.rsplit("/", 1)[0]
    ruta = f"{grupo}/{nombre}"
    return hdf_file[ruta] if ruta in hdf_file else None


def leer_radianza(hdf_file) -> Tuple[np.ndarray, dict]:
    """
    Devuelve la radianza en unidades físicas, con NaN donde no hay medición.

    Args:
        hdf_file: Archivo HDF5 abierto

    Returns:
        Tuple con (matriz float64 en nW/(cm² sr), metadatos de la lectura)
    """
    dataset = hdf_file[find_image_path(hdf_file)]
    crudo = dataset[()]
    attrs = dataset.attrs

    def atributo(nombre, defecto=None):
        valor = attrs.get(nombre, defecto)
        if isinstance(valor, np.ndarray):
            valor = valor.item() if valor.size == 1 else valor
        return valor

    relleno = atributo("_FillValue")
    minimo = atributo("valid_min")
    maximo = atributo("valid_max")
    escala = float(atributo("scale_factor", 1.0))
    desplazamiento = float(atributo("add_offset", 0.0))

    valido = np.ones(crudo.shape, dtype=bool)
    if relleno is not None:
        # El relleno de VNP46A2 es -999.9 en punto flotante; compararlo con
        # igualdad exacta es frágil, así que se usa una vecindad.
        if np.issubdtype(crudo.dtype, np.floating):
            valido &= np.abs(crudo - float(relleno)) > 1e-3
        else:
            valido &= crudo != relleno
    if minimo is not None:
        valido &= crudo >= minimo
    if maximo is not None:
        valido &= crudo <= maximo

    # VNP46A2 marca por píxel si la recuperación sirve. Sin esto, una noche
    # nublada entrega un número plausible que no es luz del suelo: en el
    # 2025-01-05 de Iztapalapa, VNP46A1 dio 62,838 y VNP46A2 dice que no hay
    # ni un píxel utilizable.
    calidad = _buscar_hermano(hdf_file, dataset, BANDERA_CALIDAD)
    if calidad is not None:
        valido &= calidad[()] == CALIDAD_ACEPTABLE

    radianza = np.full(crudo.shape, np.nan, dtype=np.float64)
    radianza[valido] = crudo[valido].astype(np.float64) * escala + desplazamiento

    unidades = atributo("units", "")
    if isinstance(unidades, bytes):
        unidades = unidades.decode(errors="replace")

    return radianza, {
        "producto": "VNP46A2" if calidad is not None else "VNP46A1",
        "calidad_aplicada": calidad is not None,
        "escala": escala,
        "desplazamiento": desplazamiento,
        "unidades": unidades,
        "invalidos": int((~valido).sum()),
        "fraccion_valida": float(valido.mean()),
    }
