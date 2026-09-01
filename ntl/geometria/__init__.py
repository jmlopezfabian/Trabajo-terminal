"""
Polígono municipal → cobertura por píxel.

Etapa que corre una vez por municipio: su resultado es estático mientras no
cambie la delimitación oficial. Produce la tabla que consume `ntl.radianza`.
"""

from .cobertura import cobertura_exacta, metricas_ponderadas, poligono_en_pixeles
from .processor import SatelliteProcessor

__all__ = [
    "SatelliteProcessor",
    "cobertura_exacta",
    "metricas_ponderadas",
    "poligono_en_pixeles",
]
