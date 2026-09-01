"""Integration tests for ntl processor with mocked download and I/O."""
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from ntl.geometria.processor import SatelliteProcessor


def _fake_cobertura(poligono):
    """Cobertura de un recorte pequeño: interior lleno y un borde a medias."""
    pesos = np.array([
        [0.0, 0.5, 0.5, 0.0],
        [0.5, 1.0, 1.0, 0.5],
        [0.0, 0.5, 0.5, 0.0],
    ])
    return pesos, 2, 3      # matriz, fila y columna de origen dentro del cuadrante


class TestSatelliteProcessor:
    def test_get_measures_returns_dict_when_mocks_ok(self, sample_hdf5_path):
        coords = np.array([[-1.05, -0.60], [-1.04, -0.59], [-1.045, -0.58], [-1.05, -0.60]])
        with patch("ntl.geometria.processor.find_file", return_value="http://example.com/file.h5"):
            with patch("ntl.geometria.processor.download_file", return_value=sample_hdf5_path):
                with patch("ntl.geometria.processor.extraer_coordenadas", return_value=coords):
                    with patch("ntl.geometria.processor.cobertura_exacta", side_effect=_fake_cobertura):
                        proc = SatelliteProcessor("Iztapalapa")
                        result = proc.get_measures("01-01-24", "h08v07", show_plots=False)
        assert result is not None
        assert isinstance(result, dict)
        assert "Fecha" in result
        assert "Cantidad_de_pixeles" in result
        assert "Media_de_radianza" in result
        assert "Suma_de_radianza" in result

    def test_get_measures_returns_none_when_find_file_fails(self):
        with patch("ntl.geometria.processor.find_file", return_value=None):
            proc = SatelliteProcessor("Iztapalapa")
            result = proc.get_measures("01-01-24", "h08v07", show_plots=False)
        assert result is None

    def test_get_measures_returns_none_when_download_fails(self):
        with patch("ntl.geometria.processor.find_file", return_value="http://example.com/file.h5"):
            with patch("ntl.geometria.processor.download_file", return_value=None):
                proc = SatelliteProcessor("Iztapalapa")
                result = proc.get_measures("01-01-24", "h08v07", show_plots=False)
        assert result is None

    def test_get_measures_returns_none_when_extraer_coordenadas_fails(self, sample_hdf5_path):
        with patch("ntl.geometria.processor.find_file", return_value="http://example.com/file.h5"):
            with patch("ntl.geometria.processor.download_file", return_value=sample_hdf5_path):
                with patch("ntl.geometria.processor.extraer_coordenadas", return_value=None):
                    proc = SatelliteProcessor("Iztapalapa")
                    result = proc.get_measures("01-01-24", "h08v07", show_plots=False)
        assert result is None

    def test_run_accumulates_results_in_dataframe(self, sample_hdf5_path):
        coords = np.array([[-1.05, -0.60], [-1.04, -0.59], [-1.045, -0.58], [-1.05, -0.60]])
        with patch("ntl.geometria.processor.find_file", return_value="http://example.com/file.h5"):
            with patch("ntl.geometria.processor.download_file", return_value=sample_hdf5_path):
                with patch("ntl.geometria.processor.extraer_coordenadas", return_value=coords):
                    with patch("ntl.geometria.processor.cobertura_exacta", side_effect=_fake_cobertura):
                        with patch("ntl.geometria.processor.os.remove"):
                            proc = SatelliteProcessor("Iztapalapa")
                            df = proc.run(["01-01-24", "02-01-24"], "h08v07", show_plots=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Fecha" in df.columns
        assert "Media_de_radianza" in df.columns

    def test_run_returns_empty_dataframe_when_no_results(self):
        with patch("ntl.geometria.processor.find_file", return_value=None):
            proc = SatelliteProcessor("Iztapalapa")
            df = proc.run(["01-01-24", "02-01-24"], "h08v07", show_plots=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
