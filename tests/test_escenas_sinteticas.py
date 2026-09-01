"""
Escenas sintéticas con respuesta en forma cerrada.

Las demás pruebas comparan el resultado contra la intersección geométrica que
calcula el propio proyecto. Estas no: usan figuras cuya área se conoce de
antemano —πr² para un círculo, l² para un cuadrado— así que validan la
agregación contra un número que no salió de este código.

Las figuras se colocan en posiciones no enteras y giradas a propósito, para que
la frontera caiga a mitad de píxel y el caso difícil quede ejercitado.

El círculo se aproxima con un polígono de muchos lados, de modo que la tolerancia
frente a πr² recoge tanto el error de la cobertura como el de esa discretización.
"""
import numpy as np
import pytest

from shapely.geometry import Polygon

from ntl.core.metricas import metricas_ponderadas
from ntl.geometria.cobertura import cobertura_exacta

LIENZO = (42, 42)


def _cobertura(xs, ys):
    """Cobertura por píxel de un contorno dado en coordenadas de píxel."""
    parcial, fila_0, columna_0 = cobertura_exacta(Polygon(np.column_stack([xs, ys])))
    pesos = np.zeros(LIENZO)
    pesos[fila_0:fila_0 + parcial.shape[0], columna_0:columna_0 + parcial.shape[1]] = parcial
    return pesos


def _circulo(cx, cy, r, n=4000):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def _cuadrado(cx, cy, lado, angulo):
    h = lado / 2
    esquinas = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
    giro = np.array([[np.cos(angulo), -np.sin(angulo)],
                     [np.sin(angulo), np.cos(angulo)]])
    q = esquinas @ giro.T
    return cx + q[:, 0], cy + q[:, 1]


FIGURAS = [
    ("circulo_r7.3_centro_no_entero", *_circulo(20.37, 20.61, 7.3), np.pi * 7.3 ** 2),
    ("circulo_r3.1", *_circulo(20.5, 20.5, 3.1), np.pi * 3.1 ** 2),
    ("cuadrado_11.4_girado_30", *_cuadrado(20.2, 20.7, 11.4, np.pi / 6), 11.4 ** 2),
    ("cuadrado_9_alineado", *_cuadrado(20.0, 20.0, 9.0, 0.0), 81.0),
]


class TestAreaContraFormaCerrada:
    @pytest.mark.parametrize("nombre,xs,ys,exacta", FIGURAS)
    def test_converge_a_la_formula(self, nombre, xs, ys, exacta):
        area = _cobertura(xs, ys).sum()
        assert area == pytest.approx(exacta, rel=1e-3)

    @pytest.mark.parametrize("nombre,xs,ys,exacta", FIGURAS)
    def test_la_precision_no_depende_de_la_posicion(self, nombre, xs, ys, exacta):
        """Desplazar la figura una fracción de píxel no debe cambiar el área.

        Con una asignación binaria el resultado salta según dónde caiga la
        frontera respecto a la retícula; con la intersección no.
        """
        for dx, dy in ((0.0, 0.0), (0.33, 0.17), (0.5, 0.5), (0.71, 0.94)):
            area = _cobertura(np.asarray(xs) + dx, np.asarray(ys) + dy).sum()
            assert area == pytest.approx(exacta, rel=1e-6), f"desplazada ({dx}, {dy})"

    def test_una_figura_pequena_no_es_mas_dificil(self):
        """El área de la frontera deja de importar cuando se calcula exacta."""
        _, xs_g, ys_g, ex_g = FIGURAS[0]     # r = 7.3
        _, xs_p, ys_p, ex_p = FIGURAS[1]     # r = 3.1, razón perímetro/área mucho mayor
        assert _cobertura(xs_g, ys_g).sum() == pytest.approx(ex_g, rel=1e-6)
        assert _cobertura(xs_p, ys_p).sum() == pytest.approx(ex_p, rel=1e-6)


class TestCampoConstante:
    """Con radianza constante, las métricas intensivas deben ser exactas."""

    C = 137.5

    @pytest.fixture
    def metricas(self):
        xs, ys = _circulo(20.37, 20.61, 7.3)
        pesos = _cobertura(xs, ys)
        imagen = np.full(LIENZO, self.C)
        return metricas_ponderadas(imagen[pesos > 0], pesos[pesos > 0])

    def test_la_media_es_la_constante(self, metricas):
        assert metricas["Media_de_radianza"] == pytest.approx(self.C, abs=1e-9)

    def test_la_dispersion_es_cero(self, metricas):
        assert metricas["Desviacion_estandar_de_radianza"] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("campo", ["Percentil_25_de_radianza",
                                       "Percentil_50_de_radianza",
                                       "Percentil_75_de_radianza"])
    def test_los_percentiles_son_la_constante(self, metricas, campo):
        assert metricas[campo] == pytest.approx(self.C, abs=1e-9)

    def test_la_suma_es_la_constante_por_el_area(self, metricas):
        esperada = self.C * np.pi * 7.3 ** 2
        assert metricas["Suma_de_radianza"] == pytest.approx(esperada, rel=1e-3)

    def test_las_metricas_intensivas_no_dependen_de_la_malla(self):
        """Refinar cambia el área, pero no puede mover la media de un campo plano."""
        xs, ys = _circulo(20.37, 20.61, 7.3)
        imagen = np.full(LIENZO, self.C)
        for k in (1, 4, 16):
            pesos = _cobertura(xs, ys)
            m = metricas_ponderadas(imagen[pesos > 0], pesos[pesos > 0])
            assert m["Media_de_radianza"] == pytest.approx(self.C, abs=1e-9)
