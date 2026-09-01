"""Integration tests for ntl processing with mocked HDF5."""
from datetime import date

import pytest

from ntl.radianza.extraccion import extract_radiance_matrix, process_image


class TestProcessImage:
    def test_returns_medicion_resultado_when_path_valid(self, sample_hdf5_path):
        # Coordinates within 10x10 fixture image
        coords = [(1, 1), (2, 1), (2, 2), (1, 2)]
        result = process_image(
            sample_hdf5_path,
            coords,
            date(2024, 1, 1),
            "Iztapalapa",
            delete_file=False,
        )
        assert result is not None
        assert result.Municipio == "Iztapalapa"
        assert result.Fecha == date(2024, 1, 1)
        assert result.Cantidad_de_pixeles == len(coords)
        assert result.Suma_de_radianza >= 0
        assert result.Media_de_radianza >= 0

    def test_returns_none_when_path_does_not_exist(self):
        result = process_image(
            "/nonexistent/path.h5",
            [(0, 0)],
            date(2024, 1, 1),
            "Test",
            delete_file=False,
        )
        assert result is None

    def test_returns_none_when_no_valid_coordinates(self, sample_hdf5_path):
        # Coordinates outside 10x10 image
        coords = [(100, 100), (101, 101)]
        result = process_image(
            sample_hdf5_path,
            coords,
            date(2024, 1, 1),
            "Test",
            delete_file=False,
        )
        assert result is None

    def test_returns_none_when_empty_coordinates(self, sample_hdf5_path):
        result = process_image(
            sample_hdf5_path,
            [],
            date(2024, 1, 1),
            "Test",
            delete_file=False,
        )
        assert result is None


class TestExtractRadianceMatrix:
    def test_returns_dict_with_matrices_when_valid(self, sample_hdf5_path):
        coords = [(1, 1), (2, 1), (2, 2), (1, 2)]
        result = extract_radiance_matrix(
            sample_hdf5_path,
            coords,
            date(2024, 1, 1),
            "iztapalapa",
        )
        assert result is not None
        assert result["municipio"] == "iztapalapa"
        assert result["fecha"] == date(2024, 1, 1)
        assert result["bbox"] == {"min_x": 1, "max_x": 2, "min_y": 1, "max_y": 2}
        assert result["rows"] == 2
        assert result["cols"] == 2
        assert len(result["radiance_matrix"]) == 2
        assert len(result["radiance_matrix"][0]) == 2
        assert len(result["municipality_mask"]) == 2
        assert len(result["municipality_mask"][0]) == 2
        assert sum(sum(row) for row in result["municipality_mask"]) == 4

    def test_returns_none_when_path_does_not_exist(self):
        result = extract_radiance_matrix(
            "/nonexistent/path.h5",
            [(0, 0)],
            date(2024, 1, 1),
            "Test",
        )
        assert result is None

    def test_returns_none_when_no_valid_coordinates(self, sample_hdf5_path):
        coords = [(100, 100), (101, 101)]
        result = extract_radiance_matrix(
            sample_hdf5_path,
            coords,
            date(2024, 1, 1),
            "Test",
        )
        assert result is None


class TestPonderacionPorCobertura:
    """La cobertura fraccional debe llegar hasta las métricas."""

    def test_la_cobertura_reduce_el_area_y_la_suma(self, sample_hdf5_path):
        completos = [(1, 1, 1.0), (2, 1, 1.0), (2, 2, 1.0), (1, 2, 1.0)]
        frontera = [(1, 1, 1.0), (2, 1, 0.5), (2, 2, 1.0), (1, 2, 0.5)]

        entero = process_image(sample_hdf5_path, completos, date(2024, 1, 1),
                               "Iztapalapa", delete_file=False)
        medio = process_image(sample_hdf5_path, frontera, date(2024, 1, 1),
                              "Iztapalapa", delete_file=False)

        assert entero.Cantidad_de_pixeles == pytest.approx(4.0)
        assert medio.Cantidad_de_pixeles == pytest.approx(3.0)
        assert medio.Suma_de_radianza < entero.Suma_de_radianza

    def test_area_no_es_el_numero_de_pixeles(self, sample_hdf5_path):
        pesos = [(1, 1, 0.25), (2, 1, 0.25)]
        r = process_image(sample_hdf5_path, pesos, date(2024, 1, 1),
                          "Iztapalapa", delete_file=False)
        assert len(pesos) == 2
        assert r.Cantidad_de_pixeles == pytest.approx(0.5)

    def test_expone_la_matriz_de_cobertura(self, sample_hdf5_path):
        pesos = [(1, 1, 1.0), (2, 1, 0.5), (2, 2, 1.0), (1, 2, 0.25)]
        r = process_image(sample_hdf5_path, pesos, date(2024, 1, 1),
                          "Iztapalapa", delete_file=False)
        assert r.Cobertura_municipio == [[1.0, 0.5], [0.25, 1.0]]
        # La máscara binaria sigue marcando qué píxeles toca el municipio
        assert r.Mascara_municipio == [[1, 1], [1, 1]]

    def test_el_formato_anterior_sigue_funcionando(self, sample_hdf5_path):
        """Una tabla sin regenerar asume cobertura 1.0 y avisa."""
        coords = [(1, 1), (2, 1), (2, 2), (1, 2)]
        r = process_image(sample_hdf5_path, coords, date(2024, 1, 1),
                          "Iztapalapa", delete_file=False)
        assert r is not None
        assert r.Cantidad_de_pixeles == pytest.approx(4.0)
