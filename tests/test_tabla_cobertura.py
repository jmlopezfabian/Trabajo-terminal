"""
La tabla de cobertura que se distribuye debe estar al día.

Es un archivo generado que vive en el repositorio, así que nada impide que
alguien cambie la geometría y olvide regenerarlo. Estas pruebas atan el archivo
al código que lo produce.
"""
import json

import numpy as np
import pytest

from ntl.core.config import PIXELES_MUNICIPIOS, RUTA_MUNICIPIOS
from ntl.core.utils import extraer_coordenadas, load_coord_data, normalize_municipio
from ntl.geometria.image_processor import pesos_municipio, recortar

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


class TestSuperficieOficial:
    """
    La tabla, convertida a km², debe reproducir la superficie real del territorio.

    Es la única prueba que valida la georreferenciación completa —esquina del
    cuadrante, resolución, transformación de coordenadas y coberturas— contra
    una referencia ajena al proyecto. Las demás dan por buena esa cadena.
    """

    # Superficie de la Ciudad de México según INEGI. Las cifras publicadas por
    # alcaldía no sirven de referencia: suman ~1461 km², 34 menos que el total
    # oficial, y las fuentes se contradicen entre sí (para Tlalpan circulan 310,
    # 312, 314.5 y 340 km²). El agregado sí es consistente.
    CDMX_KM2 = 1495.0

    @staticmethod
    def _area_celda_km2(geod, x, y, ul, res):
        """Área geodésica de una celda de la retícula sobre el elipsoide."""
        lon0, lat0 = ul[0] + x * res, ul[1] - y * res
        lon1, lat1 = lon0 + res, lat0 - res
        area, _ = geod.polygon_area_perimeter(
            [lon0, lon1, lon1, lon0], [lat0, lat0, lat1, lat1]
        )
        return abs(area) / 1e6

    @staticmethod
    def _esquina(cuadrante):
        return (-180.0 + 10.0 * int(cuadrante[1:3]), 90.0 - 10.0 * int(cuadrante[4:6]))

    def _km2(self, geod, datos):
        res = 10 / FORMA_CUADRANTE[1]
        ul = self._esquina(datos.cuadrante)
        return sum(w * self._area_celda_km2(geod, x, y, ul, res) for x, y, w in datos.pesos)

    def test_la_suma_de_las_alcaldias_es_la_superficie_de_la_cdmx(self, tabla):
        geod = pytest.importorskip("pyproj").Geod(ellps="WGS84")

        with open(RUTA_MUNICIPIOS, encoding="utf-8") as f:
            cdmx = [x["properties"]["NOMGEO"] for x in json.load(f)["features"]
                    if x["properties"]["CVE_ENT"] == "09"]

        total = sum(
            self._km2(geod, load_coord_data(normalize_municipio(nombre), PIXELES_MUNICIPIOS))
            for nombre in cdmx
        )
        assert total == pytest.approx(self.CDMX_KM2, rel=0.01)

    @pytest.mark.parametrize("nombre", ["Iztacalco", "Azcapotzalco", "Tlalpan", "Milpa Alta"])
    def test_coincide_con_el_area_geodesica_del_poligono(self, nombre):
        """Compara contra el área del mismo polígono sobre el elipsoide WGS84."""
        pyproj = pytest.importorskip("pyproj")
        geod = pyproj.Geod(ellps="WGS84")

        coordenadas = extraer_coordenadas(nombre)
        area_geodesica, _ = geod.polygon_area_perimeter(coordenadas[:, 0], coordenadas[:, 1])
        area_geodesica = abs(area_geodesica) / 1e6

        datos = load_coord_data(normalize_municipio(nombre), PIXELES_MUNICIPIOS)
        assert self._km2(geod, datos) == pytest.approx(area_geodesica, rel=1e-3)
