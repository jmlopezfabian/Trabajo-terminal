"""
Lectura de la radianza: unidades físicas y valores de relleno.

El histórico trae cuatro fechas en las que el cuadrante completo venía relleno
y 65535 se sumó como si fuera radianza; las sumas salieron 229 veces mayores
que las normales. Estas pruebas cubren esa clase de fallo.
"""
import h5py
import numpy as np
import pytest

from ntl.core.lectura import leer_radianza
from ntl.core.metricas import metricas_ponderadas

RUTA = "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m"


def _hdf5(tmp_path, datos, **attrs):
    ruta = tmp_path / "prueba.h5"
    with h5py.File(ruta, "w") as f:
        ds = f.create_dataset(RUTA, data=datos.astype(np.uint16))
        for k, v in attrs.items():
            ds.attrs[k] = v
    return ruta


class TestUnidadesFisicas:
    def test_aplica_el_factor_de_escala(self, tmp_path):
        ruta = _hdf5(tmp_path, np.array([[100, 200]]), scale_factor=0.1,
                     add_offset=0.0, units=b"nW/(cm2 sr)")
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert radianza.tolist() == [[10.0, 20.0]]
        assert meta["unidades"] == "nW/(cm2 sr)"

    def test_aplica_el_desplazamiento(self, tmp_path):
        ruta = _hdf5(tmp_path, np.array([[10]]), scale_factor=2.0, add_offset=5.0)
        with h5py.File(ruta) as f:
            radianza, _ = leer_radianza(f)
        assert radianza[0, 0] == pytest.approx(25.0)

    def test_sin_atributos_no_transforma(self, tmp_path):
        ruta = _hdf5(tmp_path, np.array([[7, 9]]))
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert radianza.tolist() == [[7.0, 9.0]]
        assert meta["escala"] == 1.0


class TestValoresDeRelleno:
    def test_el_relleno_sale_como_nan(self, tmp_path):
        ruta = _hdf5(tmp_path, np.array([[100, 65535, 300]]),
                     scale_factor=0.1, _FillValue=65535)
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert np.isnan(radianza[0, 1])
        assert radianza[0, 0] == pytest.approx(10.0)
        assert meta["invalidos"] == 1

    def test_respeta_el_rango_valido(self, tmp_path):
        ruta = _hdf5(tmp_path, np.array([[5, 50, 500]]), scale_factor=1.0,
                     valid_min=10, valid_max=100)
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert np.isnan(radianza[0, 0]) and np.isnan(radianza[0, 2])
        assert radianza[0, 1] == pytest.approx(50.0)
        assert meta["invalidos"] == 2

    def test_el_relleno_no_se_suma_como_radianza(self, tmp_path):
        """El fallo exacto de 2024-06-17, cuando el cuadrante entero era relleno."""
        ruta = _hdf5(tmp_path, np.array([[65535, 65535]]),
                     scale_factor=0.1, _FillValue=65535)
        with h5py.File(ruta) as f:
            radianza, meta = leer_radianza(f)
        assert meta["fraccion_valida"] == 0.0
        # Sin la máscara esto habría dado 13107.0 en vez de None
        assert metricas_ponderadas(radianza.ravel(), np.ones(2)) is None


class TestFraccionValida:
    def test_la_perdida_de_territorio_queda_registrada(self):
        valores = np.array([100.0, np.nan, 100.0, 100.0])
        m = metricas_ponderadas(valores, np.ones(4))
        assert m["Fraccion_valida"] == pytest.approx(0.75)
        assert m["Cantidad_de_pixeles"] == pytest.approx(3.0)

    def test_sin_perdida_la_fraccion_es_uno(self):
        m = metricas_ponderadas(np.array([1.0, 2.0]), np.ones(2))
        assert m["Fraccion_valida"] == pytest.approx(1.0)

    def test_la_suma_no_incluye_los_pixeles_sin_medicion(self):
        con_hueco = metricas_ponderadas(np.array([10.0, np.nan]), np.ones(2))
        assert con_hueco["Suma_de_radianza"] == pytest.approx(10.0)

    def test_el_area_encoge_con_los_pixeles_perdidos(self):
        """Es lo honesto: el territorio sin dato no se puede reportar como medido."""
        m = metricas_ponderadas(np.array([10.0, np.nan]), np.array([1.0, 0.5]))
        assert m["Cantidad_de_pixeles"] == pytest.approx(1.0)
        assert m["Fraccion_valida"] == pytest.approx(1.0 / 1.5)
