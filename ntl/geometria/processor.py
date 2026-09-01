import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
#configuraar backend interactivo
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, List
from ..core.config import IMAGE_PATH, find_image_path, temp_path
from ..core.lectura import leer_radianza
from ..core.models import MedicionResultado
from ..core.utils import parse_date, extraer_coordenadas, left_right_coords
from ..core.downloader import find_file, download_file
from .cobertura import cobertura_exacta, poligono_en_pixeles
from .image_processor import metricas_ponderadas

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class SatelliteProcessor:
    """
    Clase principal para procesar imágenes satelitales del producto VNP46A2.
    """
    
    def __init__(self, municipio: str):
        self.municipio = municipio
    
    def _save_plot(self, fig, date_obj: str, quadrant: str, plot_type: str = "analysis"):
        """
        Guarda una figura de matplotlib en lugar de mostrarla.
        
        Args:
            fig: Figura de matplotlib
            date_obj: Fecha del análisis
            quadrant: Cuadrante de la imagen
            plot_type: Tipo de gráfica
        """
        try:
            # Crear directorio temp si no existe
            filename = str(temp_path(f"{date_obj}_{self.municipio}_{quadrant}_{plot_type}.png"))
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada como: {filename}")
        except Exception as e:
            print(f"Error al guardar la gráfica: {e}")
        finally:
            plt.close(fig)
    
    def get_measures(self, date_str: str, quadrant: str, show_plots: bool = True) -> Optional[dict]:
        """
        Consulta, descarga y extrae las coordenadas de una imagen satelital para un día y cuadrante.
        
        Args:
            date_str: Fecha en formato dd-mm-yy
            quadrant: Cuadrante de la imagen
            show_plots: Si mostrar las gráficas de visualización

        Returns:
            Diccionario con las mediciones o None si hay error
        """
        year, day, date_obj = parse_date(date_str)

        # Buscar y descargar archivo
        h5_url = find_file(year, day, quadrant)
        if not h5_url:
            print("No se encontró el archivo.")
            return None

        save_path = str(temp_path(f"{date_obj}_{self.municipio}_{quadrant}.h5"))
        h5_save_path = download_file(h5_url, save_path)
        if not h5_save_path:
            print("Fallo la descarga del archivo.")
            return None
        
        with h5py.File(h5_save_path, "r") as hdf_file:
            left_coord, right_coord = left_right_coords(hdf_file)
            if left_coord is None or right_coord is None:
                print("No se pudieron extraer las coordenadas del archivo HDF.")
                return None
            
            # Encontrar la ruta correcta a los datos de imagen
            image_path = find_image_path(hdf_file)
            if image_path not in hdf_file:
                print(f"Error: No se encontró la ruta '{image_path}' en el archivo HDF5.")
                print("Estructura del archivo HDF5:")
                def print_structure(name, obj):
                    print(name)
                hdf_file.visititems(print_structure)
                return None
            
            image_matrix, lectura = leer_radianza(hdf_file)
            coordenadas_municipio = extraer_coordenadas(self.municipio)
            if coordenadas_municipio is None:
                print("No se pudieron extraer las coordenadas del municipio.")
                return None

            # Crear copia para visualización
            copia_imagen = np.clip(image_matrix, 0, np.percentile(image_matrix, 99))

            if show_plots:
                fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
                ax[0][0].imshow(copia_imagen)
                ax[0][0].set_title(f"Imagen completa {self.municipio} - {date_obj}")
                
            try:
                # La pertenencia de cada píxel se resuelve por intersección
                # geométrica: es el mismo cálculo que alimenta la tabla
                # precalculada del procesamiento por lotes.
                poligono = poligono_en_pixeles(
                    coordenadas_municipio, left_coord, image_matrix.shape
                )
                pesos, fila_0, columna_0 = cobertura_exacta(poligono)
                alto, ancho = pesos.shape
                imagen_recortada = image_matrix[fila_0:fila_0 + alto, columna_0:columna_0 + ancho]

                if imagen_recortada.size == 0:
                    print("La imagen recortada está vacía. Verifica las coordenadas del municipio.")
                    return None

                copia_imagen = np.clip(imagen_recortada, 0,
                                       np.nanpercentile(imagen_recortada, 99))

                # Contorno del municipio en coordenadas del recorte, para superponerlo
                contorno_x, contorno_y = poligono.exterior.coords.xy
                contorno_x = np.asarray(contorno_x) - columna_0
                contorno_y = np.asarray(contorno_y) - fila_0

                if show_plots:
                    ax[0][1].imshow(copia_imagen)
                    ax[0][1].set_title("Imagen recortada")

                    ax[1][0].imshow(copia_imagen)
                    ax[1][0].plot(contorno_x, contorno_y, 'k-', linewidth=1.5, alpha=0.9,
                                  label='Contorno del municipio')
                    ax[1][0].set_title("Recorte con el contorno superpuesto")
                    ax[1][0].legend(loc='upper right', fontsize=8)

                    im = ax[1][1].imshow(pesos, cmap="Blues", vmin=0, vmax=1)
                    ax[1][1].plot(contorno_x, contorno_y, 'k-', linewidth=1.5, alpha=0.9)
                    ax[1][1].set_title("Cobertura por píxel")
                    fig.colorbar(im, ax=ax[1][1], fraction=0.046)

                    ax[2][0].imshow(copia_imagen)
                    ax[2][0].imshow(pesos, cmap="Blues", alpha=0.45, vmin=0, vmax=1)
                    ax[2][0].plot(contorno_x, contorno_y, 'k-', linewidth=1.5, alpha=0.9)
                    ax[2][0].set_title("Radianza ponderada por cobertura")

                # Métricas ponderadas por área: invariantes al factor de escala
                metricas = metricas_ponderadas(imagen_recortada, pesos)

                if show_plots:
                    dentro = pesos > 0
                    if dentro.any():
                        ax[2][1].hist(imagen_recortada[dentro].astype(float), bins=50,
                                      weights=pesos[dentro], alpha=0.7,
                                      label='Píxeles del municipio', color='blue')
                        ax[2][1].grid(True)
                        ax[2][1].set_title("Histograma de radiación (ponderado por área)")
                        ax[2][1].legend()
                    else:
                        ax[2][1].text(0.5, 0.5, "No hay píxeles seleccionados",
                                    ha='center', va='center', transform=ax[2][1].transAxes)
                        ax[2][1].set_title("Sin datos")

                # Validar que hay píxeles para procesar
                if metricas is None:
                    print("No se encontraron píxeles dentro del área del municipio.")
                    return None

                # Crear medición usando solo MedicionResultado
                medicion = MedicionResultado(
                    Fecha=date_obj,
                    Producto=lectura['producto'],
                    Unidades_de_radianza=lectura['unidades'] or 'nW/(cm2 sr)',
                    **metricas,
                )
                if show_plots:
                    # Guardar la figura usando la función 
                    plt.show()
                    self._save_plot(plt.gcf(), date_obj, quadrant, "analysis")
                os.remove(h5_save_path)
                return medicion.model_dump()
                
            except Exception as e:
                print(f"Error durante el procesamiento de la imagen: {e}")
                return None

    def run(self, fechas: List[str], quadrant: str = "h08v07", show_plots: bool = False) -> pd.DataFrame:
        """
        Procesa múltiples fechas y retorna un dataframe con los resultados.
        
        Args:
            fechas: Lista de fechas en formato dd-mm-yy
            quadrant: Cuadrante de la imagen (por defecto h08v07)
            show_plots: Si mostrar las visualizaciones de matplotlib

        Returns:
            DataFrame con las mediciones de todas las fechas
        """
        results = []

        for fecha in fechas:
            print(f"Procesando fecha: {fecha}")
            try:
                datos = self.get_measures(fecha, quadrant, show_plots=show_plots)
                if datos:
                    results.append(datos)
                else:
                    print(f"No se pudieron obtener datos para {fecha}")
            except Exception as e:
                print(f"Error procesando {fecha}: {e}")
        
        if not results:
            print("No se obtuvieron datos para ninguna fecha.")
            return pd.DataFrame()
            
        return pd.DataFrame(results)

    def recortar_imagen_solo(self, date_str: str, quadrant: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Recorta la imagen y calcula la cobertura, sin agregar métricas.

        Args:
            date_str: Fecha en formato dd-mm-yy
            quadrant: Cuadrante de la imagen

        Returns:
            Tuple con (imagen_recortada, copia para visualización, cobertura por píxel)
        """
        year, day, date_obj = parse_date(date_str)

        h5_url = find_file(year, day, quadrant)
        if not h5_url:
            print("No se encontró el archivo.")
            return None

        save_path = str(temp_path(f"{date_obj}_{self.municipio}_{quadrant}.h5"))
        h5_save_path = download_file(h5_url, save_path)
        if not h5_save_path:
            print("Fallo la descarga del archivo.")
            return None

        with h5py.File(h5_save_path, "r") as hdf_file:
            left_coord, right_coord = left_right_coords(hdf_file)
            if left_coord is None or right_coord is None:
                print("No se pudieron extraer las coordenadas del archivo HDF.")
                return None
            
            # Encontrar la ruta correcta a los datos de imagen
            image_path = find_image_path(hdf_file)
            if image_path not in hdf_file:
                print(f"Error: No se encontró la ruta '{image_path}' en el archivo HDF5.")
                print("Estructura del archivo HDF5:")
                def print_structure(name, obj):
                    print(name)
                hdf_file.visititems(print_structure)
                return None
                
            image_matrix, _ = leer_radianza(hdf_file)
            coordenadas_municipio = extraer_coordenadas(self.municipio)

            if coordenadas_municipio is None:
                print("No se pudieron extraer las coordenadas del municipio.")
                return None

            copia_imagen = np.clip(image_matrix, 0, np.percentile(image_matrix, 99))
            
            try:
                poligono = poligono_en_pixeles(
                    coordenadas_municipio, left_coord, image_matrix.shape
                )
                pesos, fila_0, columna_0 = cobertura_exacta(poligono)
                alto, ancho = pesos.shape
                imagen_recortada = image_matrix[fila_0:fila_0 + alto, columna_0:columna_0 + ancho]

                if imagen_recortada.size == 0:
                    print("La imagen recortada está vacía. Verifica las coordenadas del municipio.")
                    return None

                copia_imagen = np.clip(imagen_recortada, 0, np.nanpercentile(imagen_recortada, 99))

                return imagen_recortada, copia_imagen, pesos
                
            except Exception as e:
                print(f"Error durante el recorte de la imagen: {e}")
                return None