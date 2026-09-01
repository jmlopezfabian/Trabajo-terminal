"""
Estadísticas de radianza ponderadas por el área que cada píxel aporta.

Única implementación de las métricas del proyecto. `geometria` la alimenta con
las coberturas que acaba de calcular y `radianza` con las que lee de la tabla
precalculada; las dos deben producir exactamente el mismo número para la misma
imagen, y con dos copias del cálculo eso no estaba garantizado.
"""
from typing import Optional

import numpy as np


def metricas_ponderadas(valores: np.ndarray, pesos: np.ndarray) -> Optional[dict]:
    """
    Agrega los valores de radianza ponderando cada uno por su cobertura.

    Args:
        valores: Radianza de cada píxel que toca el municipio
        pesos: Fracción de cada píxel que cae dentro, en (0, 1]

    Returns:
        Diccionario con los campos de MedicionResultado, o None si no hay píxeles.
        `Cantidad_de_pixeles` es un área en píxeles de la retícula original, no
        un conteo, así que es comparable entre municipios y entre corridas.
    """
    valores = np.asarray(valores, dtype=np.float64)
    pesos = np.asarray(pesos, dtype=np.float64)

    # Los píxeles sin medición llegan como NaN. Descartarlos encoge el área, que es
    # lo honesto: contarlos como radianza fue lo que arruinó cuatro fechas del
    # histórico. Fraccion_valida deja constancia de cuánto territorio se perdió.
    area_declarada = float(pesos.sum())
    medibles = np.isfinite(valores) & (pesos > 0)
    valores, pesos = valores[medibles], pesos[medibles]

    if valores.size == 0 or pesos.sum() <= 0:
        return None

    area = float(pesos.sum())
    suma = float(np.dot(pesos, valores))
    media = suma / area
    varianza = float(np.dot(pesos, (valores - media) ** 2) / area)

    p25, p50, p75 = _percentiles_ponderados(valores, pesos, area)

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
        "Fraccion_valida": (area / area_declarada) if area_declarada > 0 else 0.0,
    }


def _percentiles_ponderados(valores: np.ndarray, pesos: np.ndarray, area: float):
    """Percentiles sobre la masa de área acumulada, no sobre el conteo."""
    orden = np.argsort(valores)
    valores_ord, pesos_ord = valores[orden], pesos[orden]
    acumulada = (np.cumsum(pesos_ord) - 0.5 * pesos_ord) / area
    return np.interp([0.25, 0.50, 0.75], acumulada, valores_ord)
