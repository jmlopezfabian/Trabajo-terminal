"""
Cobertura por píxel → métricas de radianza.

Etapa que corre todos los días: aplica las coberturas ya calculadas a cada
imagen VNP46A2, descargando en paralelo y agrupando por cuadrante.
"""

from .extraccion import extract_radiance_matrix
from .lotes import SatelliteImagesAsync

__all__ = ["SatelliteImagesAsync", "extract_radiance_matrix"]
