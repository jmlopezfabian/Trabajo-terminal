"""
La tabla de cobertura que se distribuye debe estar al día.

Es un archivo generado que vive en el repositorio, así que nada impide que
alguien cambie la geometría y olvide regenerarlo. Estas pruebas atan el archivo
al código que lo produce.
"""
import json

import numpy as np
import pytest

from vnp46a1.core.config import PIXELES_MUNICIPIOS
from vnp46a1.core.utils import extraer_coordenadas, load_coord_data, normalize_municipio
from vnp46a1.geometria.image_processor import pesos_municipio, recortar

FORMA_CUADRANTE = (2400, 2400)


@pytest.fixture(scope="module")
def tabla():
    with open(PIXELES_MUNICIPIOS, encoding="utf-8") as f:
        return json.load(f)


def _esquina(cuadrante):
    return (-180.0 + 10.0 * int(cuadrante[1:3]), 90.0 - 10.0 * int(cuadrante[4:6]))


class TestFormato:
    def test_todos_los_municipios_traen_pesos(self, tabla):
        sin_pesos = [k for k, v in tabla.items() if "pesos" not in v]
        assert sin_pesos == [], (
            f"{sin_pesos} usa el formato anterior; regenera con "
            f"scripts/generar_coordenadas_pixeles.py"
        )

    def test_cada_entrada_es_x_y_cobertura(self, tabla):
        for clave, datos in tabla.items():
            for entrada in datos["pesos"]:
                assert len(entrada) == 3, f"{clave}: {entrada}"
                x, y, w = entrada
                assert isinstance(x, int) and isinstance(y, int)
                assert 0 < w <= 1, f"{clave}: cobertura fuera de (0,1]: {w}"

    def test_hay_pixeles_de_frontera(self, tabla):
        """Si todo valiera 1.0, la tabla seguiría siendo la anterior."""
        for clave, datos in tabla.items():
            parciales = [w for _, _, w in datos["pesos"] if w < 0.999]
            assert parciales, f"{clave} no tiene ningún píxel de frontera"

    def test_la_frontera_esta_cubierta_a_medias_en_promedio(self, tabla):
        """El polígono cruza esas celdas, así que cubre cerca de la mitad."""
        for clave, datos in tabla.items():
            parciales = [w for _, _, w in datos["pesos"] if w < 0.999]
            assert 0.3 < float(np.mean(parciales)) < 0.7, clave


class TestCoherenciaConLaGeometria:
    @pytest.mark.parametrize("nombre", ["Azcapotzalco", "Iztacalco", "Milpa Alta"])
    def test_el_area_coincide_con_recalcularla(self, nombre):
        """El archivo debe ser lo que produce el código de geometría hoy."""
        datos = load_coord_data(normalize_municipio(nombre), PIXELES_MUNICIPIOS)

        coordenadas = extraer_coordenadas(nombre)
        vacia = np.zeros(FORMA_CUADRANTE, dtype=np.float32)
        recorte, nx, ny = recortar(vacia, coordenadas, _esquina(datos.cuadrante), 32)
        area_recalculada = float(pesos_municipio(recorte.shape, nx, ny, 32).sum())

        assert datos.area == pytest.approx(area_recalculada, rel=1e-4)

    def test_el_area_supera_el_conteo_de_pixeles_interiores(self):
        """
        Regresión del sesgo de frontera.

        La versión anterior guardaba solo los píxeles completamente interiores,
        lo que subestimaba el área. El área nueva tiene que ser mayor que el
        número de píxeles que están cubiertos del todo.
        """
        datos = load_coord_data("azcapotzalco", PIXELES_MUNICIPIOS)
        interiores = sum(1 for _, _, w in datos.pesos if w >= 0.999)
        assert datos.area > interiores
        # El sesgo documentado para Azcapotzalco era de -19.4%
        assert (interiores - datos.area) / datos.area < -0.15
