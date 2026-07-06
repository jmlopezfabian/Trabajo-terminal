from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from datetime import date

class CoordenadasPixeles(BaseModel):
    cuadrante: str = Field(..., description="Cuadrante of the image")
    coordenadas_pixeles: List[Tuple[int, int]] = Field(..., description="Coordenadas of the pixels")

class BboxRecorte(BaseModel):
    min_x: int = Field(..., description="Minimum x coordinate")
    max_x: int = Field(..., description="Maximum x coordinate")
    min_y: int = Field(..., description="Minimum y coordinate")
    max_y: int = Field(..., description="Maximum y coordinate")

class MedicionResultado(BaseModel):
    Fecha: date = Field(..., description="Date of the measurement")
    Municipio: str = Field(..., description="Name of the municipality")
    Cantidad_de_pixeles: int = Field(..., description="Number of pixels")
    Suma_de_radianza: float = Field(..., description="Sum of the radiance")
    Media_de_radianza: float = Field(..., description="Mean of the radiance")
    Desviacion_estandar_de_radianza: float = Field(..., description="Standard deviation of the radiance")
    Maximo_de_radianza: float = Field(..., description="Maximum of the radiance")
    Minimo_de_radianza: float = Field(..., description="Minimum of the radiance")
    Percentil_25_de_radianza: float = Field(..., description="25th percentile of the radiance")
    Percentil_50_de_radianza: float = Field(..., description="50th percentile of the radiance")
    Percentil_75_de_radianza: float = Field(..., description="75th percentile of the radiance")
    Bbox: Optional[BboxRecorte] = Field(None, description="Bounding box of the cropped image in the original tile")
    Filas: Optional[int] = Field(None, description="Number of rows in the cropped radiance matrix")
    Columnas: Optional[int] = Field(None, description="Number of columns in the cropped radiance matrix")
    Matriz_de_radianza: Optional[List[List[Optional[float]]]] = Field(
        None, description="Cropped radiance matrix (Filas x Columnas); NaN/Inf as null"
    )
    Mascara_municipio: Optional[List[List[int]]] = Field(
        None, description="Binary mask over the cropped matrix: 1=municipality pixel, 0=otherwise"
    )
    