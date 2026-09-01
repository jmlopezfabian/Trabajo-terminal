from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Tuple
from datetime import date

class CoordenadasPixeles(BaseModel):
    """
    Cobertura precalculada de un municipio sobre la retícula de un cuadrante.

    Cada entrada de `pesos` es (x, y, w) con w en (0, 1]: la fracción del píxel
    que cae dentro del municipio. Los píxeles de frontera valen menos de 1, que
    es justo lo que un conjunto de coordenadas no podía expresar.
    """

    cuadrante: str = Field(..., description="Cuadrante of the image")
    pesos: List[Tuple[int, int, float]] = Field(
        ..., description="(x, y, coverage) per pixel; coverage in (0, 1]"
    )

    @model_validator(mode="before")
    @classmethod
    def _acepta_formato_anterior(cls, data):
        """
        Lee el formato previo, en el que solo había coordenadas.

        Un archivo viejo sigue cargando, con cobertura 1.0 en cada píxel. Eso
        reproduce el comportamiento anterior, que subestimaba el área del
        municipio porque descartaba la franja de frontera entera; regenera el
        archivo con scripts/generar_coordenadas_pixeles.py para corregirlo.
        """
        if isinstance(data, dict) and "pesos" not in data and "coordenadas_pixeles" in data:
            data = dict(data)
            data["pesos"] = [(x, y, 1.0) for x, y in data.pop("coordenadas_pixeles")]
        return data

    @property
    def coordenadas_pixeles(self) -> List[Tuple[int, int]]:
        """Solo las coordenadas, sin cobertura. Para máscaras y recortes."""
        return [(x, y) for x, y, _ in self.pesos]

    @property
    def area(self) -> float:
        """Área del municipio en píxeles de la retícula original."""
        return sum(w for _, _, w in self.pesos)

class BboxRecorte(BaseModel):
    min_x: int = Field(..., description="Minimum x coordinate")
    max_x: int = Field(..., description="Maximum x coordinate")
    min_y: int = Field(..., description="Minimum y coordinate")
    max_y: int = Field(..., description="Maximum y coordinate")

class MedicionResultado(BaseModel):
    Fecha: date = Field(..., description="Date of the measurement")
    Municipio: Optional[str] = Field(None, description="Name of the municipality")
    Cantidad_de_pixeles: float = Field(
        ...,
        description=(
            "Area of the municipality in original pixels: the sum of the per-pixel "
            "coverage weights. Invariant to the scale factor, unlike a subpixel count."
        ),
    )
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
    Cobertura_municipio: Optional[List[List[float]]] = Field(
        None,
        description=(
            "Fraction of each pixel of the cropped matrix that falls inside the "
            "municipality, in [0, 1]. Weight these to reproduce the aggregate metrics."
        ),
    )
    