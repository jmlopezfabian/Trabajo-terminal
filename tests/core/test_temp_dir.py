"""Tests for the staging directory used to download HDF5 files.

It used to be the literal relative path "../temp", so where a several-hundred-MB
download landed depended on the process's working directory. These tests pin
the replacement down: absolute, configurable, and independent of the cwd.
"""
import importlib
import re
from pathlib import Path

import pytest

from ntl.core import config


def _reload_config(monkeypatch, temp_dir=None):
    """Re-import config with NTL_TEMP_DIR set (or unset)."""
    if temp_dir is None:
        monkeypatch.delenv("NTL_TEMP_DIR", raising=False)
        monkeypatch.delenv("VNP46A1_TEMP_DIR", raising=False)
    else:
        monkeypatch.setenv("NTL_TEMP_DIR", str(temp_dir))
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def restore_config():
    """Leave the module as we found it for the rest of the suite."""
    yield
    importlib.reload(config)


def test_temp_dir_is_absolute_by_default(monkeypatch):
    cfg = _reload_config(monkeypatch)
    assert cfg.TEMP_DIR.is_absolute()


def test_temp_dir_honours_the_environment(monkeypatch, tmp_path):
    target = tmp_path / "staging"
    cfg = _reload_config(monkeypatch, target)
    assert cfg.TEMP_DIR == target.resolve()


def test_temp_dir_does_not_depend_on_the_cwd(monkeypatch, tmp_path):
    """The actual bug: the same code resolved to different directories."""
    cfg = _reload_config(monkeypatch)
    from_here = cfg.TEMP_DIR

    elsewhere = tmp_path / "some" / "other" / "place"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    cfg = _reload_config(monkeypatch)

    assert cfg.TEMP_DIR == from_here


def test_temp_path_creates_the_directory_on_first_use(monkeypatch, tmp_path):
    target = tmp_path / "not-yet-there"
    cfg = _reload_config(monkeypatch, target)
    assert not target.exists()

    path = cfg.temp_path("VNP46A1_2024-01-15_h08v07.h5")

    assert target.is_dir()
    assert path.parent == target.resolve()
    assert path.name == "VNP46A1_2024-01-15_h08v07.h5"
    assert path.is_absolute()


def test_download_and_cleanup_agree_on_the_directory(monkeypatch, tmp_path):
    """Downloads and cleanup used to be able to point at different places."""
    target = tmp_path / "staging"
    _reload_config(monkeypatch, target)

    from ntl.radianza import lotes as sa

    importlib.reload(sa)

    staged = sa.temp_path("2024-01-15_h08v07.h5")
    staged.write_bytes(b"not really hdf5")
    assert staged.exists()

    monkeypatch.chdir(tmp_path)  # cleanup must not care where we are
    sa.cleanup_temp_files()

    assert not staged.exists()


def test_no_relative_parent_paths_remain():
    """Guard against a cwd-relative path being assigned again.

    Matches assignments such as `temp_dir = "../temp"`, not prose mentioning
    the old path in a comment or docstring. Cubre cualquier ruta al directorio
    padre, no solo "../temp": el mismo fallo reapareció en "../data" y la
    versión anterior de esta prueba no lo veía.
    """
    import ntl

    # rglob, no glob: la mitad de geometría vivía en otro paquete y por eso
    # conservó el "../temp" durante todo el tiempo que este test estuvo en verde.
    source_dir = Path(ntl.__file__).parent
    pattern = re.compile(r"""=\s*f?["'][^"']*\.\./""")
    offenders = [
        str(path.relative_to(source_dir))
        for path in source_dir.rglob("*.py")
        if pattern.search(path.read_text())
    ]
    assert offenders == []


def test_acepta_el_nombre_anterior_de_la_variable(monkeypatch, tmp_path):
    """Un despliegue con VNP46A1_TEMP_DIR configurado no debe romperse."""
    monkeypatch.delenv("NTL_TEMP_DIR", raising=False)
    monkeypatch.setenv("VNP46A1_TEMP_DIR", str(tmp_path / "antiguo"))
    cfg = importlib.reload(config)
    assert cfg.TEMP_DIR == (tmp_path / "antiguo").resolve()


def test_el_nombre_nuevo_tiene_precedencia(monkeypatch, tmp_path):
    monkeypatch.setenv("VNP46A1_TEMP_DIR", str(tmp_path / "antiguo"))
    monkeypatch.setenv("NTL_TEMP_DIR", str(tmp_path / "nuevo"))
    cfg = importlib.reload(config)
    assert cfg.TEMP_DIR == (tmp_path / "nuevo").resolve()
