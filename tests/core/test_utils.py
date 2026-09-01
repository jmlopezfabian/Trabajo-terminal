"""Unit tests for ntl utils (pure functions and I/O with mocks)."""
import json
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import numpy as np
import pytest

from ntl.core.utils import (
    normalize_municipio,
    parse_date,
    distancia_puntos,
    polygon_centroid,
    extraer_coordenadas,
    load_coord_data,
    left_right_coords,
    cuadrante_de_coordenadas,
    esquina_superior_izquierda,
    verificar_georreferencia,
)


# --- Pure functions (no I/O) ---


class TestNormalizeMunicipio:
    def test_lowercase(self):
        assert normalize_municipio("Iztapalapa") == "iztapalapa"

    def test_removes_accents(self):
        assert normalize_municipio("Álvaro") == "alvaro"
        assert normalize_municipio("México") == "mexico"
        assert "ó" not in normalize_municipio("ó")


class TestParseDate:
    def test_parse_valid_date(self):
        year, day, date_obj = parse_date("01-01-24")
        assert year == 2024
        assert day == "001"
        assert date_obj.year == 2024 and date_obj.month == 1 and date_obj.day == 1

    def test_parse_mid_year(self):
        year, day, date_obj = parse_date("15-06-23")
        assert year == 2023
        assert day == "166"
        assert date_obj.month == 6 and date_obj.day == 15

    def test_day_is_zero_padded_to_three_digits(self):
        """El archivo de LAADS publica el día con tres dígitos.

        Las dos versiones del procesamiento diferían aquí: una devolvía el
        entero y construía URLs como /2024/1/ en vez de /2024/001/.
        """
        for fecha in ("01-01-24", "05-01-24", "31-12-24"):
            _, day, _ = parse_date(fecha)
            assert isinstance(day, str) and len(day) == 3

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            parse_date("2024-01-01")


class TestDistanciaPuntos:
    def test_same_point(self):
        assert distancia_puntos((1.0, 2.0), (1.0, 2.0)) == 0.0

    def test_horizontal_distance(self):
        d = distancia_puntos((0.0, 0.0), (3.0, 0.0))
        assert d == pytest.approx(3.0)

    def test_diagonal_distance(self):
        d = distancia_puntos((0.0, 0.0), (3.0, 4.0))
        assert d == pytest.approx(5.0)


class TestPolygonCentroid:
    def test_square_centroid(self):
        # Unit square
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        cx, cy = polygon_centroid(vertices)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)

    def test_triangle_centroid(self):
        vertices = [(0, 0), (2, 0), (1, 2)]
        cx, cy = polygon_centroid(vertices)
        assert cx == pytest.approx(1.0)
        assert cy == pytest.approx(2.0 / 3.0)


# --- I/O with mocks ---


class TestExtraerCoordenadas:
    def test_returns_coordinates_when_found(self, municipios_json_path):
        with patch("ntl.core.utils.RUTA_MUNICIPIOS", municipios_json_path):
            coords = extraer_coordenadas("Iztapalapa")
        assert coords is not None
        assert isinstance(coords, np.ndarray)
        assert len(coords) > 0

    def test_returns_none_when_not_found(self, municipios_json_path):
        with patch("ntl.core.utils.RUTA_MUNICIPIOS", municipios_json_path):
            coords = extraer_coordenadas("MunicipioInexistente")
        assert coords is None

    def test_returns_none_on_invalid_json(self):
        with patch("ntl.core.utils.RUTA_MUNICIPIOS", "/nonexistent/path.json"):
            with patch("builtins.open", mock_open(read_data="invalid {")):
                coords = extraer_coordenadas("Iztapalapa")
        assert coords is None


class TestLoadCoordData:
    def test_loads_coordenadas_pixeles(self, tmp_path):
        path = tmp_path / "coords.json"
        path.write_text(
            json.dumps({
                "Iztapalapa": {
                    "cuadrante": "h08v07",
                    "coordenadas_pixeles": [(10, 20), (11, 21)],
                }
            }),
            encoding="utf-8",
        )
        obj = load_coord_data("Iztapalapa", str(path))
        assert obj.cuadrante == "h08v07"
        assert obj.coordenadas_pixeles == [(10, 20), (11, 21)]


class TestLeftRightCoords:
    def test_extracts_upper_left_lower_right(self):
        meta = (
            "UpperLeftPointMtrs=(-1111950.519723,-555975.259861)\n"
            "LowerRightMtrs=(-1000755.467751,-666300.346481)\n"
        )
        # Value that supports .tobytes().decode("utf-8") as in utils
        meta_val = np.array(meta, dtype="S" + str(len(meta) + 1)).flat[0]
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = meta_val
        mock_file = MagicMock()
        mock_file.__getitem__.return_value = mock_ds
        left, right = left_right_coords(mock_file)
        assert left is not None
        assert right is not None
        assert left[0] == pytest.approx(-1.111950519723)
        assert right[0] == pytest.approx(-1.000755467751)

    def test_returns_none_when_metadata_missing(self):
        mock_file = MagicMock()
        mock_file.__getitem__ = MagicMock(side_effect=KeyError("no metadata"))
        left, right = left_right_coords(mock_file)
        assert left is None
        assert right is None


def _hdf_con_metadata(meta: str):
    """Archivo HDF5 simulado que solo sabe devolver su StructMetadata.0."""
    meta_val = np.array(meta, dtype="S" + str(len(meta) + 1)).flat[0]
    mock_ds = MagicMock()
    mock_ds.__getitem__.return_value = meta_val
    mock_file = MagicMock()
    mock_file.__getitem__.return_value = mock_ds
    return mock_file


# StructMetadata.0 de un granulo real: VNP46A1.A2024015.h08v07.002. Las cifras
# vienen en grados por un millon, no en metros, pese al nombre del campo: el
# producto se publica en una reticula lat/lon lineal, no sinusoidal.
METADATA_H08V07 = (
    'GridName="VIIRS_Grid_DNB_2d"\n'
    "XDim=2400\n"
    "YDim=2400\n"
    "UpperLeftPointMtrs=(-100000000.000000,20000000.000000)\n"
    "LowerRightMtrs=(-90000000.000000,10000000.000000)\n"
    "Projection=HE5_GCTP_GEO\n"
    "GridOrigin=HE5_HDFE_GD_UL\n"
)


class TestEsquinaSuperiorIzquierda:
    def test_deriva_el_origen_del_identificador(self):
        assert esquina_superior_izquierda("h08v07") == (-100.0, 20.0)
        assert esquina_superior_izquierda("h00v00") == (-180.0, 90.0)
        assert esquina_superior_izquierda("h35v17") == (170.0, -80.0)

    def test_coincide_con_lo_que_declara_un_granulo_real(self):
        """El supuesto del que cuelga toda la tabla, contra el archivo."""
        ul, _ = left_right_coords(_hdf_con_metadata(METADATA_H08V07))
        assert ul == pytest.approx(esquina_superior_izquierda("h08v07"))

    @pytest.mark.parametrize("malo", ["h8v7", "H08V07", "h08v7", "", "v07h08"])
    def test_rechaza_identificadores_mal_formados(self, malo):
        """Antes se troceaba la cadena por posicion, sin comprobar nada."""
        with pytest.raises(ValueError):
            esquina_superior_izquierda(malo)


class TestCuadranteDeCoordenadas:
    def test_deduce_el_cuadrante_de_la_cdmx(self):
        coords = np.array([[-99.22, 19.51], [-99.14, 19.45], [-99.20, 19.48]])
        assert cuadrante_de_coordenadas(coords) == "h08v07"

    def test_deduce_el_cuadrante_de_monterrey(self):
        coords = np.array([[-100.43, 25.80], [-100.18, 25.50]])
        assert cuadrante_de_coordenadas(coords) == "h07v06"

    def test_es_inverso_de_la_esquina(self):
        for cuadrante in ["h08v07", "h07v06", "h00v00", "h20v11"]:
            lon, lat = esquina_superior_izquierda(cuadrante)
            dentro = np.array([[lon + 1.0, lat - 1.0], [lon + 9.0, lat - 9.0]])
            assert cuadrante_de_coordenadas(dentro) == cuadrante

    def test_falla_si_el_poligono_cruza_de_cuadrante(self):
        """
        Componer dos imagenes no esta implementado; devolver una sola
        recortaria el municipio por la mitad sin avisar.
        """
        coords = np.array([[-100.4, 25.8], [-99.6, 25.5]])
        with pytest.raises(ValueError, match="cruza de cuadrante"):
            cuadrante_de_coordenadas(coords)


class TestVerificarGeorreferencia:
    """
    La comprobacion que faltaba: que la imagen este donde el codigo supone.

    Las coberturas se precalculan sobre un origen derivado del identificador
    del cuadrante, sin abrir el archivo. Nada confrontaba ese supuesto con la
    realidad, y un desplazamiento de un solo pixel mueve la media de radianza
    municipal un 4.6% en la mediana de las alcaldias.
    """

    def test_acepta_un_granulo_bien_georreferenciado(self):
        ul = verificar_georreferencia(
            _hdf_con_metadata(METADATA_H08V07), "h08v07", (2400, 2400)
        )
        assert ul == pytest.approx((-100.0, 20.0))

    def test_rechaza_el_cuadrante_equivocado(self):
        with pytest.raises(ValueError, match="Georreferencia incoherente"):
            verificar_georreferencia(
                _hdf_con_metadata(METADATA_H08V07), "h08v06", (2400, 2400)
            )

    def test_detecta_un_desplazamiento_de_medio_pixel(self):
        """
        El caso que motiva la comprobacion: no un cuadrante cambiado, sino un
        cambio de convencion de esquina a centro de pixel. Son ~230 m que
        ninguna prueba de area veria, porque validan contra el mismo supuesto.
        """
        medio_pixel = 0.5 * (10 / 2400) * 1e6
        meta = METADATA_H08V07.replace(
            "UpperLeftPointMtrs=(-100000000.000000,20000000.000000)",
            f"UpperLeftPointMtrs=({-100000000.0 + medio_pixel:.6f},"
            f"{20000000.0 - medio_pixel:.6f})",
        )
        with pytest.raises(ValueError, match="Georreferencia incoherente"):
            verificar_georreferencia(_hdf_con_metadata(meta), "h08v07", (2400, 2400))

    def test_rechaza_una_extension_que_no_es_de_diez_grados(self):
        """
        La esquina puede coincidir y la resolucion no. Entonces el error crece
        con la distancia al origen, que es peor: los pixeles del centro del
        cuadrante quedarian bien y los del borde no.
        """
        meta = METADATA_H08V07.replace(
            "LowerRightMtrs=(-90000000.000000,10000000.000000)",
            "LowerRightMtrs=(-95000000.000000,15000000.000000)",
        )
        with pytest.raises(ValueError, match="Extension inesperada|Extensión inesperada"):
            verificar_georreferencia(_hdf_con_metadata(meta), "h08v07", (2400, 2400))

    def test_sin_metadatos_avisa_pero_no_rompe(self, capsys):
        """
        Un archivo sin StructMetadata legible no es motivo para tirar el
        procesamiento del dia: se sigue con el supuesto de siempre, avisando.
        """
        mock_file = MagicMock()
        mock_file.__getitem__ = MagicMock(side_effect=KeyError("no metadata"))
        ul = verificar_georreferencia(mock_file, "h08v07", (2400, 2400))
        assert ul == esquina_superior_izquierda("h08v07")
        assert "no se pudo" in capsys.readouterr().out.lower()
