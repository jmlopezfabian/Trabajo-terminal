"""Unit tests for satellite_sync image_processor (pure numpy/geometry)."""
import numpy as np
import pytest

from satellite_sync.image_processor import (
    aumentar_imagen,
    recortar_imagen,
    completar_bordes,
    get_pixeles,
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
