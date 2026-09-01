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

from .config import find_image_path


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
        valido &= crudo != relleno
    if minimo is not None:
        valido &= crudo >= minimo
    if maximo is not None:
        valido &= crudo <= maximo

    radianza = np.full(crudo.shape, np.nan, dtype=np.float64)
    radianza[valido] = crudo[valido].astype(np.float64) * escala + desplazamiento

    unidades = atributo("units", "")
    if isinstance(unidades, bytes):
        unidades = unidades.decode(errors="replace")

    return radianza, {
        "escala": escala,
        "desplazamiento": desplazamiento,
        "unidades": unidades,
        "invalidos": int((~valido).sum()),
        "fraccion_valida": float(valido.mean()),
    }
