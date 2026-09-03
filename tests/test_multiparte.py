"""
Municipios que no son un solo polígono.

Un municipio puede tener islas o exclaves —GeoJSON lo publica entonces como
MultiPolygon— y puede tener huecos, cuando otro municipio queda enclavado dentro
de su territorio. El lector de límites no expresaba ninguna de las dos cosas:
devolvía ``coordinates[0]``, que en un polígono con hueco es solo el anillo
exterior, y en un MultiPolygon es la lista de anillos de la primera parte.

Ninguno de los dos casos fallaba, que es lo que los hacía peligrosos:

- Con un hueco, el enclave se contaba como territorio propio y el área salía de
  más.
- Con varias partes, ``coordenadas[:, 0]`` dejaba de ser "las longitudes" y
  pasaba a ser "el primer anillo entero", así que el cuadrante se deducía de una
  mezcla de longitudes y latitudes. Sin excepción y sin aviso.

Las figuras del fixture tienen área conocida de antemano —rectángulos de lados
enteros en grados— así que se validan contra un número que no salió de este
código. La retícula de prueba es de 120 px por cuadrante: 12 píxeles por grado.
"""
from datetime import date
from unittest.mock import patch

import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon

from ntl.core.metricas import metricas_ponderadas
from ntl.core.utils import cuadrantes_de_coordenadas
from ntl.geometria.cobertura import cobertura_exacta
from ntl.geometria.mosaico import (
    cobertura_por_cuadrante,
    partes,
    poligono_en_pixeles_globales,
)
from ntl.radianza.extraccion import process_image_mosaico

from test_multicuadrante import FORMA, crear_cuadrante, radianza_sintetica

PX_POR_GRADO = FORMA[0] / 10.0  # 12


@pytest.fixture(autouse=True)
def limites_multiparte(fixtures_dir, monkeypatch):
    """Apunta el lector de límites al fixture de este archivo."""
    monkeypatch.setattr("ntl.core.utils.RUTA_MUNICIPIOS",
                        str(fixtures_dir / "municipios_multiparte.json"))


@pytest.fixture
def geometria():
    from ntl.core.utils import extraer_geometria
    return extraer_geometria


def area_px(piezas):
    return sum(w for pesos in piezas.values() for _, _, w in pesos)


class TestLecturaDeLimites:
    """Lo que el lector puede y no puede expresar."""

    def test_lee_las_dos_partes_de_un_municipio_con_isla(self, geometria):
        g = geometria("Continental con isla")
        assert isinstance(g, MultiPolygon)
        assert len(g.geoms) == 2

    def test_lee_el_hueco_de_un_municipio_con_enclave(self, geometria):
        g = geometria("Con enclave")
        assert isinstance(g, Polygon)
        assert len(g.interiors) == 1
        # 2x2 grados de exterior menos 1x1 de enclave.
        assert g.area == pytest.approx(3.0)

    def test_el_lector_de_un_solo_contorno_falla_en_vez_de_devolver_una_parte(self):
        """
        Devolver la primera parte como si fuera el municipio entero es
        exactamente lo que hacía antes, y sin decirlo.
        """
        from ntl.core.utils import extraer_coordenadas

        with pytest.raises(ValueError, match="2 partes"):
            extraer_coordenadas("Continental con isla")

    def test_el_lector_de_un_solo_contorno_falla_ante_un_hueco(self):
        from ntl.core.utils import extraer_coordenadas

        with pytest.raises(ValueError, match="hueco"):
            extraer_coordenadas("Con enclave")

    def test_un_municipio_simple_se_sigue_leyendo_como_siempre(self):
        from ntl.core.utils import extraer_coordenadas

        coords = extraer_coordenadas("Simple")
        assert coords.shape[1] == 2
        assert float(coords[:, 0].min()) == pytest.approx(-95.0)
        assert float(coords[:, 1].max()) == pytest.approx(15.0)


class TestAreaConPartesYHuecos:
    """El área es la que dice la geometría, no la del primer anillo."""

    def test_la_isla_suma_al_area(self, geometria):
        piezas = cobertura_por_cuadrante(geometria("Continental con isla"), FORMA)
        # 1x1 grado de continente (144 px) + 0.5x0.5 de isla (36 px).
        assert area_px(piezas) == pytest.approx(180.0, rel=1e-9)

    def test_ignorar_la_isla_habria_perdido_una_quinta_parte(self, geometria):
        """Lo que costaba quedarse con la primera parte."""
        completo = area_px(cobertura_por_cuadrante(
            geometria("Continental con isla"), FORMA))
        solo_continental = area_px(cobertura_por_cuadrante(
            partes(geometria("Continental con isla"))[0], FORMA))
        assert solo_continental / completo == pytest.approx(0.8, rel=1e-9)

    def test_el_enclave_resta_del_area(self, geometria):
        piezas = cobertura_por_cuadrante(geometria("Con enclave"), FORMA)
        # 2x2 grados (576 px) menos el enclave de 1x1 (144 px).
        assert area_px(piezas) == pytest.approx(432.0, rel=1e-9)

    def test_ignorar_el_enclave_habria_inflado_el_area_un_tercio(self, geometria):
        """
        El anillo exterior solo —lo que devolvía el lector anterior— cuenta el
        enclave como territorio propio.
        """
        con_hueco = area_px(cobertura_por_cuadrante(geometria("Con enclave"), FORMA))
        sin_hueco = area_px(cobertura_por_cuadrante(
            Polygon(geometria("Con enclave").exterior), FORMA))
        assert sin_hueco / con_hueco == pytest.approx(576 / 432, rel=1e-9)

    def test_ningun_pixel_del_enclave_entra_en_la_cobertura(self, geometria):
        """El hueco no es cobertura parcial: es cero."""
        piezas = cobertura_por_cuadrante(geometria("Con enclave"), FORMA)
        pesos = {(x, y): w for p in piezas.values() for x, y, w in p}
        # Centro del enclave, en píxeles globales del cuadrante h08v07.
        centro_x = int((-94.0 + 180.0) * PX_POR_GRADO) % FORMA[1]
        centro_y = int((90.0 - 14.0) * PX_POR_GRADO) % FORMA[0]
        assert (centro_x, centro_y) not in pesos

    def test_las_partes_que_se_tocan_no_duplican_su_frontera(self, geometria):
        """
        Dos mitades pegadas comparten la columna de píxeles de su frontera. Las
        áreas se suman —es lo que haría el municipio entero— y ninguna cobertura
        se pasa de 1.
        """
        piezas = cobertura_por_cuadrante(geometria("Dos partes que se tocan"), FORMA)
        # Las dos mitades juntas son un rectángulo de 1 x 0.5 grados: 72 px.
        assert area_px(piezas) == pytest.approx(72.0, rel=1e-9)
        assert all(0 < w <= 1.0 for p in piezas.values() for _, _, w in p)

    def test_el_area_de_las_partes_suma_la_del_conjunto(self, geometria):
        """Invariante general, sin apoyarse en las medidas del fixture."""
        for nombre in ["Continental con isla", "Con enclave",
                       "Isla en otro cuadrante", "Dos partes que se tocan"]:
            g = geometria(nombre)
            esperada = g.area * PX_POR_GRADO ** 2
            assert area_px(cobertura_por_cuadrante(g, FORMA)) == pytest.approx(
                esperada, rel=1e-9), nombre


class TestFronterasAMitadDePixel:
    """Las figuras del fixture caen en bordes de píxel; desplazarlas no debe importar."""

    @pytest.mark.parametrize("nombre", ["Continental con isla", "Con enclave"])
    @pytest.mark.parametrize("dx,dy", [(0.0, 0.0), (0.017, 0.0), (0.0, 0.031),
                                       (0.041, 0.023)])
    def test_el_area_no_depende_de_donde_caiga_la_reticula(self, geometria, nombre, dx, dy):
        g = geometria(nombre)
        esperada = g.area * PX_POR_GRADO ** 2
        movida = translate(g, xoff=dx, yoff=dy)
        assert area_px(cobertura_por_cuadrante(movida, FORMA)) == pytest.approx(
            esperada, rel=1e-6)

    def test_la_frontera_del_enclave_produce_coberturas_parciales(self, geometria):
        """
        Con el enclave a mitad de píxel tiene que haber celdas cubiertas en
        parte por dentro del hueco: si el hueco se estuviera resolviendo por
        celda entera, no las habría.
        """
        movida = translate(geometria("Con enclave"), xoff=0.041, yoff=0.023)
        piezas = cobertura_por_cuadrante(movida, FORMA)
        parciales = [w for p in piezas.values() for _, _, w in p if 0 < w < 0.999]
        assert len(parciales) > 40


class TestMultiparteYMulticuadrante:
    """Los dos casos difíciles a la vez."""

    def test_una_isla_al_otro_lado_de_la_esquina_da_dos_piezas_de_cuatro(self, geometria):
        """
        Con figuras conexas, una envolvente de cuatro cuadrantes obliga a tocar
        al menos tres. Un municipio con islas rompe esa regla: dos partes en
        diagonal tienen la caja en cuatro cuadrantes y territorio en dos, y las
        otras dos imágenes no hay que descargarlas.
        """
        g = geometria("Isla en otro cuadrante")
        assert len(cuadrantes_de_coordenadas(g)) == 4

        piezas = cobertura_por_cuadrante(g, FORMA)
        assert list(piezas) == ["h08v07", "h09v08"]
        assert area_px(piezas) == pytest.approx(72.0, rel=1e-9)

    def test_cada_parte_queda_en_su_propio_cuadrante(self, geometria):
        piezas = cobertura_por_cuadrante(geometria("Isla en otro cuadrante"), FORMA)
        assert area_px({"h08v07": piezas["h08v07"]}) == pytest.approx(36.0, rel=1e-9)
        assert area_px({"h09v08": piezas["h09v08"]}) == pytest.approx(36.0, rel=1e-9)

    def test_la_cobertura_se_recorre_parte_por_parte_no_por_la_envolvente(self, geometria):
        """
        La envolvente de un municipio con una isla lejana incluye todo lo que hay
        en medio, y el recorrido de `cobertura_exacta` visita cada celda de la
        envolvente que se le pasa. Se llama una vez por parte, no una vez por el
        conjunto: con una isla a 80 km la diferencia son minutos de recorrer mar.
        """
        g = geometria("Isla en otro cuadrante")
        with patch("ntl.geometria.mosaico.cobertura_exacta",
                   side_effect=cobertura_exacta) as espia:
            cobertura_por_cuadrante(g, FORMA)

        assert espia.call_count == 2
        for llamada in espia.call_args_list:
            minx, miny, maxx, maxy = llamada.args[0].bounds
            # Ninguna llamada recibe la envolvente entera (2 grados = 24 px).
            assert maxx - minx == pytest.approx(6.0)
            assert maxy - miny == pytest.approx(6.0)


class TestProcesamientoDeUnMunicipioConIslas:
    """De extremo a extremo: dos islas, dos imágenes, una sola agregación."""

    @pytest.fixture
    def mosaico(self, tmp_path):
        return {c: crear_cuadrante(tmp_path, c) for c in ["h08v07", "h09v08"]}

    def oraculo(self, g):
        """Métricas sobre la retícula global, parte por parte, sin cuadrantes."""
        valores, cobertura = [], []
        for parte in partes(poligono_en_pixeles_globales(g, FORMA)):
            pesos, fila_0, columna_0 = cobertura_exacta(parte)
            filas, columnas = np.nonzero(pesos)
            for f, c in zip(filas, columnas):
                valores.append(radianza_sintetica(int(c) + columna_0, int(f) + fila_0))
                cobertura.append(round(float(pesos[f, c]), 6))
        return metricas_ponderadas(np.array(valores), np.array(cobertura))

    def test_las_metricas_son_las_de_las_dos_islas_juntas(self, geometria, mosaico):
        g = geometria("Isla en otro cuadrante")
        piezas = cobertura_por_cuadrante(g, FORMA)

        r = process_image_mosaico(mosaico, piezas, date(2024, 1, 1), "insular",
                                  delete_files=False)
        esperado = self.oraculo(g)

        assert r is not None
        assert r.Cuadrantes == ["h08v07", "h09v08"]
        assert r.Cantidad_de_pixeles == pytest.approx(esperado["Cantidad_de_pixeles"], rel=1e-12)
        assert r.Media_de_radianza == pytest.approx(esperado["Media_de_radianza"], rel=1e-12)
        # La mediana de las dos islas juntas no es el promedio de sus medianas:
        # solo coincide porque los pares (radianza, cobertura) se agregan una vez.
        assert r.Percentil_50_de_radianza == pytest.approx(
            esperado["Percentil_50_de_radianza"], rel=1e-12)
        assert r.Fraccion_valida == pytest.approx(1.0)

    def test_el_recorte_abarca_las_dos_islas_y_el_mar_de_en_medio(self, geometria, mosaico):
        """
        El bounding box es uno solo, así que incluye el hueco entre las islas.
        La máscara distingue: solo los píxeles del municipio están marcados.
        """
        g = geometria("Isla en otro cuadrante")
        piezas = cobertura_por_cuadrante(g, FORMA)
        r = process_image_mosaico(mosaico, piezas, date(2024, 1, 1), "insular",
                                  delete_files=False)

        mascara = np.array(r.Mascara_municipio)
        assert mascara.sum() == sum(len(p) for p in piezas.values())
        # El recorte es mucho mayor que el territorio: entre las dos islas hay
        # mar, y eso es correcto, siempre que no cuente como municipio.
        assert mascara.size > 3 * mascara.sum()
        assert np.array(r.Cobertura_municipio).sum() == pytest.approx(
            r.Cantidad_de_pixeles, rel=1e-9)

    def test_si_falta_la_imagen_de_una_isla_se_declara(self, geometria, mosaico):
        g = geometria("Isla en otro cuadrante")
        piezas = cobertura_por_cuadrante(g, FORMA)
        rutas = dict(mosaico)
        rutas["h09v08"] = None

        r = process_image_mosaico(rutas, piezas, date(2024, 1, 1), "insular",
                                  delete_files=False)
        assert r.Cuadrantes_faltantes == ["h09v08"]
        # Las dos islas son iguales, así que se pierde exactamente la mitad.
        assert r.Fraccion_valida == pytest.approx(0.5, rel=1e-9)
