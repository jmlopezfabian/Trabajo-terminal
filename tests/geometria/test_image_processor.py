"""Unit tests for ntl image_processor (pure numpy/geometry)."""
import numpy as np
import pytest

from ntl.geometria.image_processor import (
    aumentar_imagen,
    recortar_imagen,
    completar_bordes,
    get_pixeles,
    pesos_municipio,
    metricas_ponderadas,
)


class TestAumentarImagen:
    def test_factor_one_equivalent(self):
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = aumentar_imagen(img, 1)
        np.testing.assert_array_almost_equal(out, img)

    def test_factor_two_doubles_shape(self):
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = aumentar_imagen(img, 2)
        assert out.shape == (4, 4)
        assert out[0, 0] == 1.0
        assert out[1, 1] == 1.0

    def test_factor_three(self):
        img = np.ones((2, 2))
        out = aumentar_imagen(img, 3)
        assert out.shape == (6, 6)


class TestRecortarImagen:
    def test_recorte_basic(self):
        # Image 10x10, upper_left in same coordinate system as coords
        image = np.random.rand(10, 10).astype(np.float32)
        # Municipio bbox in "pixel-like" coords: we need coords in same units as upper_left
        # upper_left (0,0) in 10x10 image with resolution 1.0 -> pixels 0-10
        resolucion = 10 / 10  # 1.0
        upper_left = (0.0, 10.0)  # typical lat/lon style, y decreases down
        # Coords that map to pixels (1,1) to (5,5) in image
        coords = np.array([
            [1.0, 9.0],
            [5.0, 9.0],
            [5.0, 5.0],
            [1.0, 5.0],
            [1.0, 9.0],
        ])
        recortada, nx, ny = recortar_imagen(image, coords, upper_left, factor_escala=1)
        assert recortada.size > 0
        assert len(nx) == len(coords)
        assert len(ny) == len(coords)

    def test_recorte_with_scale_factor(self):
        image = np.ones((20, 20), dtype=np.float32)
        upper_left = (0.0, 20.0)
        coords = np.array([[2.0, 18.0], [6.0, 18.0], [6.0, 14.0], [2.0, 14.0], [2.0, 18.0]])
        recortada, nx, ny = recortar_imagen(image, coords, upper_left, factor_escala=2)
        assert recortada.size > 0
        assert recortada.shape[0] >= 8  # 4 rows * 2
        assert recortada.shape[1] >= 8


class TestCompletarBordes:
    def test_consecutive_points_unchanged(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.0, 0.0])
        bordes = completar_bordes(x, y)
        assert len(bordes) >= 3
        assert (0, 0) in bordes
        assert (2, 0) in bordes

    def test_gap_filled(self):
        x = np.array([0.0, 10.0])  # gap of 10
        y = np.array([0.0, 0.0])
        bordes = completar_bordes(x, y)
        assert len(bordes) >= 2


def _borde_rectangular(x0, y0, x1, y1):
    """Borde cerrado de un rectangulo, un pixel por posicion."""
    borde = []
    for x in range(x0, x1 + 1):
        borde += [(x, y0), (x, y1)]
    for y in range(y0, y1 + 1):
        borde += [(x0, y), (x1, y)]
    return sorted(set(borde))


class TestGetPixeles:
    def test_interior_de_un_rectangulo(self):
        img = np.zeros((7, 7))
        bordes = _borde_rectangular(1, 1, 5, 5)
        pixels = set(get_pixeles(img, bordes))
        esperado = {(x, y) for x in range(2, 5) for y in range(2, 5)}
        assert pixels == esperado

    def test_no_incluye_el_borde_ni_el_exterior(self):
        img = np.zeros((7, 7))
        bordes = _borde_rectangular(1, 1, 5, 5)
        pixels = set(get_pixeles(img, bordes))
        assert not (pixels & set(bordes))
        assert (0, 0) not in pixels
        assert (6, 6) not in pixels

    def test_sin_interior_devuelve_lista_vacia(self):
        """Un borde de 3x3 solo encierra un pixel; uno de 2x2 no encierra nada."""
        img = np.zeros((5, 5))
        bordes = _borde_rectangular(1, 1, 2, 2)
        assert get_pixeles(img, bordes) == []

    def test_regiones_interiores_desconectadas(self):
        """Dos cuartos separados dentro de la imagen: ambos deben detectarse.

        El BFS sembrado en un centroide solo alcanzaba uno de ellos; por eso
        existia el paso extra de "pixeles huerfanos".
        """
        img = np.zeros((7, 12))
        bordes = _borde_rectangular(1, 1, 4, 5) + _borde_rectangular(7, 1, 10, 5)
        pixels = set(get_pixeles(img, bordes))
        esperado = ({(x, y) for x in range(2, 4) for y in range(2, 5)} |
                    {(x, y) for x in range(8, 10) for y in range(2, 5)})
        assert pixels == esperado

    def test_poligono_concavo_no_desborda(self):
        """Poligono en 'U': su centroide cae fuera, pero el interior es correcto.

        Regresion del metodo anterior, que sembraba el flood fill en el centroide
        y terminaba inundando toda la imagen.
        """
        img = np.zeros((13, 13))
        verts = [(1, 1), (4, 1), (4, 8), (8, 8), (8, 1), (11, 1), (11, 11), (1, 11), (1, 1)]
        bordes = []
        for (x0, y0), (x1, y1) in zip(verts, verts[1:]):
            pasos = max(abs(x1 - x0), abs(y1 - y0))
            for t in range(pasos + 1):
                p = (round(x0 + (x1 - x0) * t / pasos), round(y0 + (y1 - y0) * t / pasos))
                if p not in bordes:
                    bordes.append(p)

        pixels = set(get_pixeles(img, bordes))
        # El hueco de la "U" (columnas 5-7, filas 0-7) queda fuera del municipio
        assert not any((x, y) in pixels for x in range(5, 8) for y in range(0, 8))
        # Los brazos de la "U" y su base si son interiores
        assert (2, 3) in pixels and (9, 3) in pixels and (5, 9) in pixels
        # Nada del marco exterior de la imagen
        assert not any((x, 0) in pixels or (x, 12) in pixels for x in range(13))

    def test_devuelve_tuplas_de_enteros(self):
        img = np.zeros((6, 6))
        bordes = _borde_rectangular(1, 1, 4, 4)
        pixels = get_pixeles(img, bordes)
        assert pixels
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pixels)
        assert all(isinstance(c, int) for p in pixels for c in p)

    def test_bordes_fuera_de_la_imagen_se_ignoran(self):
        img = np.zeros((5, 5))
        bordes = _borde_rectangular(1, 1, 3, 3) + [(-4, 2), (99, 2), (2, -7)]
        pixels = set(get_pixeles(img, bordes))
        assert pixels == {(2, 2)}


def _cuadrado(lado, k=1, origen=2.0):
    """Vértices de un cuadrado cerrado, ya escalados por k."""
    x0 = origen * k
    x1 = (origen + lado) * k
    x = np.array([x0, x1, x1, x0, x0])
    y = np.array([x0, x0, x1, x1, x0])
    return x, y


class TestCompletarBordesIndependienteDeK:
    def test_arista_vertical_no_produce_nan(self):
        # dx = 0 hacía pendiente infinita y int(NaN) reventaba
        x = np.array([5.0, 5.0])
        y = np.array([0.0, 40.0])
        bordes = completar_bordes(x, y)
        assert all(isinstance(c, int) for p in bordes for c in p)
        assert len(bordes) >= 40

    def test_arista_larga_no_deja_huecos(self):
        # Con 100 puntos fijos, una arista de 400 px quedaba con huecos
        x = np.linspace(0.0, 400.0, 2)
        y = np.array([0.0, 300.0])
        bordes = sorted(completar_bordes(x, y))
        for (x0, y0), (x1, y1) in zip(bordes, bordes[1:]):
            assert max(abs(x1 - x0), abs(y1 - y0)) <= 1

    @pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32])
    def test_borde_sella_el_poligono_para_cualquier_k(self, k):
        x, y = _cuadrado(6, k)
        alto = ancho = 10
        pesos = pesos_municipio((alto, ancho), x, y, k, peso_borde=0.0)
        # Si el trazo tuviera huecos, el relleno se fugaría y el área sería 0
        assert pesos.sum() > 0


class TestPesosMunicipio:
    @pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
    def test_area_converge_al_valor_geometrico(self, k):
        lado = 6
        x, y = _cuadrado(lado, k)
        pesos = pesos_municipio((10, 10), x, y, k)
        assert pesos.sum() == pytest.approx(lado * lado, rel=0.05)

    def test_pesos_acotados_en_cero_uno(self):
        k = 8
        x, y = _cuadrado(6, k)
        pesos = pesos_municipio((10, 10), x, y, k)
        assert pesos.min() >= 0.0
        assert pesos.max() <= 1.0

    def test_forma_es_la_del_recorte_no_la_de_la_malla_fina(self):
        k = 8
        x, y = _cuadrado(6, k)
        pesos = pesos_municipio((10, 10), x, y, k)
        assert pesos.shape == (10, 10)

    def test_peso_borde_acota_el_area_por_ambos_lados(self):
        k = 4
        x, y = _cuadrado(6, k)
        sin_borde = pesos_municipio((10, 10), x, y, k, peso_borde=0.0).sum()
        con_borde = pesos_municipio((10, 10), x, y, k, peso_borde=1.0).sum()
        assert sin_borde <= 36 <= con_borde

    def test_mascara_no_se_corrompe_en_arreglos_grandes(self):
        # Regresión: con numpy 1.x sobre Python 3.14, `~mascara` sobreescribía
        # la máscara en sitio a partir de 256 KB e inflaba el área al doble.
        # La malla fina supera ese umbral: 35*16 x 35*16 = 313600 bytes.
        k = 16
        x, y = _cuadrado(28, k, origen=3.0)
        pesos = pesos_municipio((35, 35), x, y, k)
        assert pesos.sum() == pytest.approx(28 * 28, rel=0.02)


class TestMetricasPonderadas:
    def test_invariante_al_factor_de_escala(self):
        rng = np.random.default_rng(0)
        imagen = rng.uniform(10, 1000, size=(12, 12))
        referencia = None
        for k in [1, 2, 4, 8, 16]:
            x, y = _cuadrado(6, k)
            pesos = pesos_municipio(imagen.shape, x, y, k)
            m = metricas_ponderadas(imagen, pesos)
            if referencia is None:
                referencia = m
            else:
                assert m["Suma_de_radianza"] == pytest.approx(referencia["Suma_de_radianza"], rel=0.05)
                assert m["Media_de_radianza"] == pytest.approx(referencia["Media_de_radianza"], rel=0.05)
                assert m["Cantidad_de_pixeles"] == pytest.approx(referencia["Cantidad_de_pixeles"], rel=0.05)

    def test_suma_no_se_infla_con_k(self):
        imagen = np.full((12, 12), 100.0)
        sumas = []
        for k in [1, 4, 16]:
            x, y = _cuadrado(6, k)
            pesos = pesos_municipio(imagen.shape, x, y, k)
            sumas.append(metricas_ponderadas(imagen, pesos)["Suma_de_radianza"])
        # Antes esto crecía como k^2: 3600, 57600, 921600
        assert max(sumas) / min(sumas) < 1.1
        assert sumas[0] == pytest.approx(3600, rel=0.05)

    def test_cantidad_de_pixeles_es_area_no_conteo_de_subpixeles(self):
        imagen = np.ones((12, 12))
        k = 8
        x, y = _cuadrado(6, k)
        pesos = pesos_municipio(imagen.shape, x, y, k)
        m = metricas_ponderadas(imagen, pesos)
        assert m["Cantidad_de_pixeles"] == pytest.approx(36, rel=0.05)

    def test_municipio_vacio_devuelve_none(self):
        imagen = np.ones((5, 5))
        assert metricas_ponderadas(imagen, np.zeros((5, 5))) is None
