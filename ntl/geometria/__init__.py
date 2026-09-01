"""
Polígono municipal → cobertura por píxel.

Etapa que corre una vez por municipio: su resultado es estático mientras no
cambie la delimitación oficial. Produce la tabla que consume `ntl.radianza`.
"""

from .cobertura import cobertura_exacta, poligono_en_pixeles
from .image_processor import (
    aumentar_imagen,
    completar_bordes,
    get_pixeles,
    metricas_ponderadas,
    pesos_municipio,
    recortar,
    recortar_imagen,
)
from .processor import SatelliteProcessor

__all__ = [
    "SatelliteProcessor",
    "cobertura_exacta",
    "poligono_en_pixeles",
    "aumentar_imagen",
    "completar_bordes",
    "get_pixeles",
    "metricas_ponderadas",
    "pesos_municipio",
    "recortar",
    "recortar_imagen",
]
