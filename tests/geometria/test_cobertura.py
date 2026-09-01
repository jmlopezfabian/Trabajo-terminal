"""Cobertura exacta y métricas ponderadas."""
import numpy as np
import pytest
from shapely.geometry import Polygon

from ntl.geometria.cobertura import cobertura_exacta, metricas_ponderadas, poligono_en_pixeles


LIENZO = (14, 14)


def _cuadrado(lado, origen=2.0):
    """Cuadrado alineado con la retícula, en coordenadas de píxel."""
    x0, x1 = origen, origen + lado
    return Polygon([(x0, x0), (x1, x0), (x1, x1), (x0, x1)])


def _pesos_de(poligono, forma):
    """Cobertura recolocada en un lienzo de la forma pedida."""
    parcial, fila_0, columna_0 = cobertura_exacta(poligono)
    pesos = np.zeros(forma)
    pesos[fila_0:fila_0 + parcial.shape[0], columna_0:columna_0 + parcial.shape[1]] = parcial
    return pesos


class TestCoberturaExacta:
    def test_un_cuadrado_alineado_da_pesos_enteros(self):
        pesos, _, _ = cobertura_exacta(_cuadrado(6))
        assert pesos.sum() == pytest.approx(36.0)
        assert set(np.unique(pesos)) <= {0.0, 1.0}

    def test_un_cuadrado_desplazado_reparte_la_frontera(self):
        pesos, _, _ = cobertura_exacta(_cuadrado(6, origen=2.5))
        assert pesos.sum() == pytest.approx(36.0)
        parciales = pesos[(pesos > 0) & (pesos < 1)]
        assert parciales.size > 0
        assert parciales.max() < 1.0

    def test_los_pesos_estan_acotados(self):
        pesos, _, _ = cobertura_exacta(_cuadrado(5.3, origen=1.7))
        assert pesos.min() >= 0.0 and pesos.max() <= 1.0

    def test_el_origen_situa_el_recorte_en_la_reticula(self):
        pesos, fila_0, columna_0 = cobertura_exacta(_cuadrado(4, origen=10.0))
        assert (fila_0, columna_0) == (10, 10)
        assert pesos.shape == (4, 4)

    def test_el_poligono_se_transforma_a_coordenadas_de_pixel(self):
        """Un grado de la retícula equivale a 240 píxeles."""
        coordenadas = np.array([[-100.0, 20.0], [-99.0, 20.0], [-99.0, 19.0], [-100.0, 19.0]])
        poligono = poligono_en_pixeles(coordenadas, (-100.0, 20.0), (2400, 2400))
        minx, miny, maxx, maxy = poligono.bounds
        assert (minx, miny) == pytest.approx((0.0, 0.0))
        assert (maxx, maxy) == pytest.approx((240.0, 240.0))


class TestMetricasPonderadas:
    """
    Envoltura sobre `core.metricas` para cobertura en forma de matriz.

    El cálculo estadístico se prueba en tests/core/test_metricas.py; aquí solo
    interesa que la envoltura seleccione los píxeles correctos y delegue bien.
    """

    def test_pondera_por_la_cobertura_de_cada_pixel(self):
        pesos = np.array([[1.0, 0.5], [0.0, 0.25]])
        imagen = np.array([[100.0, 100.0], [999.0, 100.0]])
        m = metricas_ponderadas(imagen, pesos)
        assert m["Cantidad_de_pixeles"] == pytest.approx(1.75)
        assert m["Suma_de_radianza"] == pytest.approx(175.0)

    def test_ignora_los_pixeles_de_cobertura_nula(self):
        """El píxel exterior vale 999 y no debe influir en ninguna métrica."""
        pesos = np.array([[1.0, 0.0]])
        m = metricas_ponderadas(np.array([[10.0, 999.0]]), pesos)
        assert m["Maximo_de_radianza"] == 10.0
        assert m["Media_de_radianza"] == pytest.approx(10.0)

    def test_el_area_es_la_suma_de_coberturas_no_un_conteo(self):
        pesos = np.array([[0.25, 0.25, 0.25, 0.25]])
        m = metricas_ponderadas(np.ones((1, 4)), pesos)
        assert pesos.size == 4
        assert m["Cantidad_de_pixeles"] == pytest.approx(1.0)

    def test_sobre_un_cuadrado_reproduce_su_area(self):
        pesos = _pesos_de(_cuadrado(6, origen=2.5), LIENZO)
        m = metricas_ponderadas(np.full(LIENZO, 4.0), pesos)
        assert m["Cantidad_de_pixeles"] == pytest.approx(36.0)
        assert m["Suma_de_radianza"] == pytest.approx(144.0)
        assert m["Media_de_radianza"] == pytest.approx(4.0)

    def test_sin_cobertura_devuelve_none(self):
        assert metricas_ponderadas(np.ones((3, 3)), np.zeros((3, 3))) is None
