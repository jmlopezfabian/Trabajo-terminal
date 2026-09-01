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
from ntl.core.utils import (
    esquina_superior_izquierda,
    extraer_coordenadas,
    load_coord_data,
    normalize_municipio,
)
from ntl.geometria.image_processor import pesos_municipio, recortar

FORMA_CUADRANTE = (2400, 2400)


@pytest.fixture(scope="module")
def shapely_geometry():
    return pytest.importorskip("shapely.geometry")


@pytest.fixture(scope="module")
def tabla():
    with open(PIXELES_MUNICIPIOS, encoding="utf-8") as f:
        return json.load(f)




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
        recorte, nx, ny = recortar(vacia, coordenadas, esquina_superior_izquierda(datos.cuadrante), 32)
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

    def _km2(self, geod, datos):
        res = 10 / FORMA_CUADRANTE[1]
        ul = esquina_superior_izquierda(datos.cuadrante)
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


RESOLUCION = 10 / FORMA_CUADRANTE[1]


class TestOracleDeShapely:
    """
    Los pesos deben coincidir con el área exacta de intersección polígono-píxel.

    `pesos_municipio` decide la geometría sobre una submalla de 32x32 y asigna
    0.5 a los subpíxeles que el trazo del borde atraviesa. Es un heurístico
    razonable, no una fórmula de área, así que conviene atarlo a la única
    respuesta que no admite discusión: recortar el polígono contra cada celda.

    ATENCIÓN a lo que esta prueba NO valida. Construye las celdas con el mismo
    origen que usa el pipeline, así que confirma el *rasterizado*, no el
    *anclaje* de la retícula. Si la esquina del cuadrante estuviera mal, oracle
    y pipeline coincidirían y los dos estarían desplazados. De eso se encarga
    `utils.verificar_georreferencia`, que contrasta el origen contra el
    StructMetadata.0 del archivo, y `scripts/sensibilidad_desplazamiento.py`,
    que mide lo que costaría equivocarse.
    """

    # Tolerancias: el área agregada es lo que alimenta las métricas, y ahí el
    # heurístico del borde se cancela entre celdas. Por píxel suelto puede
    # desviarse más, porque el trazo reparte la celda a ojo.
    TOLERANCIA_AREA = 0.005      # 0.5% del área del municipio
    TOLERANCIA_PIXEL = 0.05      # cobertura, en [0,1]

    # Una celda que el oracle da como tocada pero el pipeline no lista solo
    # importa si aporta área apreciable; por debajo de esto es una esquina.
    COBERTURA_DESPRECIABLE = 0.01

    @staticmethod
    def _cobertura_exacta(poligono, cuadrante, celdas):
        """Fracción de cada celda que cubre el polígono, con recorte exacto."""
        from shapely.geometry import box

        ulx, uly = esquina_superior_izquierda(cuadrante)
        exacta = {}
        for x, y in celdas:
            lon = ulx + x * RESOLUCION
            lat = uly - y * RESOLUCION
            celda = box(lon, lat - RESOLUCION, lon + RESOLUCION, lat)
            area = poligono.intersection(celda).area
            if area > 0:
                exacta[(x, y)] = area / (RESOLUCION * RESOLUCION)
        return exacta

    @pytest.mark.parametrize(
        "nombre",
        [
            "Iztacalco",              # pequeño y compacto
            "Azcapotzalco",
            "Álvaro Obregón",         # borde muy recortado
            "Milpa Alta",             # grande, con tramos casi rectos
            "Monterrey",              # otro cuadrante, latitud distinta
        ],
    )
    def test_los_pesos_reproducen_el_area_exacta(self, nombre, shapely_geometry):
        datos = load_coord_data(normalize_municipio(nombre), PIXELES_MUNICIPIOS)
        poligono = shapely_geometry.Polygon(extraer_coordenadas(nombre))
        if not poligono.is_valid:
            poligono = poligono.buffer(0)

        pesos = {(x, y): w for x, y, w in datos.pesos}
        # Un anillo de holgura alrededor: así el oracle puede encontrar celdas
        # que el pipeline se hubiera dejado fuera.
        xs = [x for x, _ in pesos]
        ys = [y for _, y in pesos]
        celdas = [
            (x, y)
            for y in range(min(ys) - 1, max(ys) + 2)
            for x in range(min(xs) - 1, max(xs) + 2)
        ]
        exacta = self._cobertura_exacta(poligono, datos.cuadrante, celdas)

        area_pipeline = sum(pesos.values())
        area_exacta = sum(exacta.values())
        assert area_pipeline == pytest.approx(area_exacta, rel=self.TOLERANCIA_AREA)

        peor = max(
            (abs(pesos.get(c, 0.0) - exacta.get(c, 0.0)), c)
            for c in set(pesos) | set(exacta)
        )
        assert peor[0] < self.TOLERANCIA_PIXEL, f"{nombre}: peor celda {peor[1]}"

    @pytest.mark.parametrize("nombre", ["Iztacalco", "Álvaro Obregón", "Monterrey"])
    def test_no_falta_ni_sobra_ningun_pixel(self, nombre, shapely_geometry):
        """El conjunto de celdas tocadas debe ser el mismo, no solo su suma."""
        datos = load_coord_data(normalize_municipio(nombre), PIXELES_MUNICIPIOS)
        poligono = shapely_geometry.Polygon(extraer_coordenadas(nombre))
        if not poligono.is_valid:
            poligono = poligono.buffer(0)

        pesos = {(x, y): w for x, y, w in datos.pesos}
        xs = [x for x, _ in pesos]
        ys = [y for _, y in pesos]
        celdas = [
            (x, y)
            for y in range(min(ys) - 1, max(ys) + 2)
            for x in range(min(xs) - 1, max(xs) + 2)
        ]
        exacta = self._cobertura_exacta(poligono, datos.cuadrante, celdas)

        faltan = [c for c, w in exacta.items() if c not in pesos and w > self.COBERTURA_DESPRECIABLE]
        sobran = [c for c in pesos if c not in exacta]
        assert not faltan, f"{nombre}: el pipeline se deja {len(faltan)} celdas, p.ej. {faltan[:5]}"
        assert not sobran, f"{nombre}: el pipeline inventa {len(sobran)} celdas, p.ej. {sobran[:5]}"


class TestParticionDeFronteras:
    """
    Dos municipios vecinos se reparten el píxel que comparten; no lo duplican.

    Cada uno se rasteriza por su cuenta y sin saber del otro, así que nada
    obliga a que las coberturas de una misma celda sumen uno. Si el trazo del
    borde se contara entero a los dos lados, la frontera aparecería dos veces:
    el área agregada crecería y la radianza de las zonas limítrofes pesaría el
    doble. Esta prueba es lo que descarta ese doble conteo.

    El margen sale del heurístico: sobre la celda que el borde cruza cada lado
    reclama media traza, y las dos mitades no tienen por qué encajar al
    milímetro.
    """

    EXCESO_TOLERADO = 0.02

    @staticmethod
    def _cobertura_acumulada(tabla, cuadrante):
        acumulada = {}
        for clave, datos in tabla.items():
            if datos["cuadrante"] != cuadrante:
                continue
            for x, y, w in datos["pesos"]:
                celda = (x, y)
                total, quienes = acumulada.get(celda, (0.0, []))
                acumulada[celda] = (total + w, quienes + [(clave, w)])
        return acumulada

    def test_hay_pixeles_compartidos_que_examinar(self, tabla):
        """Si no hubiera fronteras comunes, la prueba siguiente no probaría nada."""
        acumulada = self._cobertura_acumulada(tabla, "h08v07")
        compartidos = [c for c, (_, quienes) in acumulada.items() if len(quienes) > 1]
        assert len(compartidos) > 100, len(compartidos)

    def test_ningun_pixel_se_reparte_mas_de_una_vez(self, tabla):
        acumulada = self._cobertura_acumulada(tabla, "h08v07")

        excedidos = [
            (celda, total, quienes)
            for celda, (total, quienes) in acumulada.items()
            if total > 1.0 + self.EXCESO_TOLERADO
        ]
        assert not excedidos, (
            f"{len(excedidos)} píxeles con cobertura total > 1; el peor es "
            f"{max(excedidos, key=lambda e: e[1])}"
        )

    def test_el_reparto_de_la_frontera_no_infla_el_area(self, tabla):
        """
        El exceso agregado sobre todos los píxeles compartidos debe ser ínfimo.

        Un solo píxel puede pasarse de 1 sin que importe; lo que rompería las
        métricas es que el sesgo apunte siempre en la misma dirección.
        """
        acumulada = self._cobertura_acumulada(tabla, "h08v07")
        compartidos = {c: t for c, (t, quienes) in acumulada.items() if len(quienes) > 1}

        exceso = sum(max(0.0, t - 1.0) for t in compartidos.values())
        assert exceso / len(compartidos) < 0.001, (
            f"exceso medio {exceso / len(compartidos):.4f} px sobre "
            f"{len(compartidos)} píxeles compartidos"
        )
