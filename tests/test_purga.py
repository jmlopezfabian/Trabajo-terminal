"""Purga de registros contaminados por valores de relleno."""
import importlib.util
import os

import pandas as pd
import pytest

_RUTA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "purgar_registros_invalidos.py")
_spec = importlib.util.spec_from_file_location("purgar", _RUTA)
purgar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(purgar)


def _df(maximos, medias=None):
    n = len(maximos)
    return pd.DataFrame({
        "Fecha": ["2024-01-01"] * n,
        "Municipio": [f"m{i}" for i in range(n)],
        "Maximo_de_radianza": maximos,
        "Media_de_radianza": medias if medias is not None else maximos,
        "Suma_de_radianza": [x * 10 for x in maximos],
    })


class TestDeteccion:
    def test_detecta_el_relleno_sin_escalar(self):
        marcados = purgar.marcar_contaminados(_df([1000.0, 65535.0, 2000.0]))
        assert marcados.tolist() == [False, True, False]

    def test_detecta_el_relleno_ya_escalado(self):
        """Una serie con scale_factor aplicado trae el relleno como 6553.5."""
        marcados = purgar.marcar_contaminados(_df([100.0, 6553.5, 200.0]))
        assert marcados.tolist() == [False, True, False]

    def test_no_marca_valores_altos_legitimos(self):
        """El máximo real más alto del histórico ronda 37000 DN."""
        marcados = purgar.marcar_contaminados(_df([37193.0, 6000.0, 65000.0]))
        assert not marcados.any()

    def test_detecta_la_contaminacion_parcial(self):
        """Unos pocos píxeles de relleno: el máximo delata, la media no."""
        marcados = purgar.marcar_contaminados(_df([65535.0], medias=[820.0]))
        assert marcados.tolist() == [True]

    def test_la_precedencia_de_operadores_no_rompe_la_condicion(self):
        """`|` liga más fuerte que `<`: sin paréntesis esto reventaba."""
        marcados = purgar.marcar_contaminados(_df([65535.0, 6553.5, 1.0]))
        assert marcados.tolist() == [True, True, False]


class TestEscritura:
    def test_no_sobrescribe_la_entrada(self, tmp_path, capsys, monkeypatch):
        entrada = tmp_path / "datos.csv"
        _df([1000.0, 65535.0]).to_csv(entrada, index=False)
        monkeypatch.setattr("sys.argv", ["purgar", str(entrada), "--salida", str(entrada)])
        assert purgar.main() == 1
        assert "no puede ser el archivo de entrada" in capsys.readouterr().out

    def test_escribe_solo_los_registros_sanos(self, tmp_path, monkeypatch):
        entrada = tmp_path / "datos.csv"
        salida = tmp_path / "limpio.csv"
        _df([1000.0, 65535.0, 2000.0]).to_csv(entrada, index=False)
        monkeypatch.setattr("sys.argv", ["purgar", str(entrada), "--salida", str(salida)])
        assert purgar.main() == 0

        limpio = pd.read_csv(salida)
        assert len(limpio) == 2
        assert limpio["Maximo_de_radianza"].max() == 2000.0
        # El original se queda como estaba
        assert len(pd.read_csv(entrada)) == 3

    def test_solo_reportar_no_escribe(self, tmp_path, monkeypatch):
        entrada = tmp_path / "datos.csv"
        _df([1000.0, 65535.0]).to_csv(entrada, index=False)
        monkeypatch.setattr("sys.argv", ["purgar", str(entrada), "--solo-reportar"])
        assert purgar.main() == 0
        assert not (tmp_path / "datos_limpio.csv").exists()
