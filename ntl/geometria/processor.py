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
from .image_processor import (
    recortar,
    recortar_imagen,
    completar_bordes,
    get_pixeles,
    pesos_municipio,
    metricas_ponderadas,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class SatelliteProcessor:
    """
    Clase principal para procesar imágenes satelitales del producto VNP46A1.
    """
    
    def __init__(self, municipio: str, factor_escala: int = 1):
        self.municipio = municipio
        self.factor_escala = factor_escala
    
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
    
    def get_measures(self, date_str: str, quadrant: str, show_plots: bool = True, factor_escala: int = None) -> Optional[dict]:
        """
        Consulta, descarga y extrae las coordenadas de una imagen satelital para un día y cuadrante.
        
        Args:
            date_str: Fecha en formato dd-mm-yy
            quadrant: Cuadrante de la imagen
            show_plots: Si mostrar las gráficas de visualización
            factor_escala: Factor de escala para aumentar la resolución de la imagen (por defecto usa el del constructor)
            
        Returns:
            Diccionario con las mediciones o None si hay error
        """
        year, day, date_obj = parse_date(date_str)
        
        # Usar el factor de escala pasado como parámetro o el del constructor
        escala_a_usar = factor_escala if factor_escala is not None else self.factor_escala

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
                
            # Recortar imagen. El recorte se queda en resolución original: el
            # factor de escala solo viaja en las coordenadas del borde.
            try:
                imagen_recortada, nuevos_x, nuevos_y = recortar(
                    image_matrix, coordenadas_municipio, left_coord, escala_a_usar
                )

                # Validar que la imagen recortada no esté vacía
                if imagen_recortada.size == 0:
                    print("La imagen recortada está vacía. Verifica las coordenadas del municipio.")
                    return None

                copia_imagen = np.clip(imagen_recortada, 0, np.percentile(imagen_recortada, 99))

                # Para graficar, las coordenadas vuelven a la malla original
                grafica_x = nuevos_x / escala_a_usar
                grafica_y = nuevos_y / escala_a_usar

                if show_plots:
                    ax[0][1].imshow(imagen_recortada)
                    ax[0][1].set_title("Imagen recortada")

                # Preparar coordenadas de bordes incompletos para visualización (sin modificar la imagen)
                bordes_incompletos_x = []
                bordes_incompletos_y = []
                for i in range(len(nuevos_y)):
                    if (0 <= int(grafica_y[i]) < imagen_recortada.shape[0] and
                        0 <= int(grafica_x[i]) < imagen_recortada.shape[1]):
                        bordes_incompletos_x.append(int(grafica_x[i]))
                        bordes_incompletos_y.append(int(grafica_y[i]))

                if show_plots:
                    ax[1][0].imshow(copia_imagen)
                    if bordes_incompletos_x:
                        ax[1][0].plot(bordes_incompletos_x, bordes_incompletos_y, 'k-', linewidth=1.5, alpha=0.8, label='Bordes')
                        ax[1][0].scatter(bordes_incompletos_x, bordes_incompletos_y, c='red', s=1, alpha=0.6)
                    ax[1][0].set_title("Imagen con bordes incompletos")

                # Preparar coordenadas de bordes completos para visualización.
                # Solo se trazan si hay que graficar: la medición no los necesita,
                # pesos_municipio los calcula por su cuenta.
                bordes_completos_x = []
                bordes_completos_y = []
                if show_plots:
                    for coordenada in completar_bordes(nuevos_x, nuevos_y):
                        bx, by = coordenada[0] / escala_a_usar, coordenada[1] / escala_a_usar
                        if (0 <= by < imagen_recortada.shape[0] and
                            0 <= bx < imagen_recortada.shape[1]):
                            bordes_completos_x.append(bx)
                            bordes_completos_y.append(by)

                if show_plots:
                    ax[1][1].imshow(copia_imagen)
                    if bordes_completos_x:
                        # Dibujar bordes como línea cerrada
                        if len(bordes_completos_x) > 1:
                            ax[1][1].plot(bordes_completos_x + [bordes_completos_x[0]], 
                                         bordes_completos_y + [bordes_completos_y[0]], 
                                         'k-', linewidth=2, alpha=0.9, label='Bordes completos')
                        ax[1][1].scatter(bordes_completos_x, bordes_completos_y, c='red', s=2, alpha=0.7)
                    ax[1][1].set_title("Imagen con bordes completos")


                # Aquí entra el producto Kronecker: la malla fina se construye
                # solo para decidir qué fracción de cada píxel original cae
                # dentro, y se colapsa de inmediato a una matriz de pesos del
                # tamaño del recorte. La radianza nunca se replica.
                pesos = pesos_municipio(
                    imagen_recortada.shape, nuevos_x, nuevos_y, escala_a_usar
                )

                if show_plots:
                    ax[2][0].imshow(copia_imagen)
                    # Dibujar bordes
                    if bordes_completos_x:
                        if len(bordes_completos_x) > 1:
                            ax[2][0].plot(bordes_completos_x + [bordes_completos_x[0]],
                                         bordes_completos_y + [bordes_completos_y[0]],
                                         'k-', linewidth=2, alpha=0.9, label='Bordes')
                    # Sombrear cada píxel según la fracción que aporta al municipio
                    ax[2][0].imshow(pesos, cmap="Blues", alpha=0.45)
                    ax[2][0].set_title(f"Cobertura por píxel (k={escala_a_usar})")
                    ax[2][0].legend(loc='upper right', fontsize=8)

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

    def run(self, fechas: List[str], quadrant: str = "h08v07", show_plots: bool = False, factor_escala: int = None) -> pd.DataFrame:
        """
        Procesa múltiples fechas y retorna un dataframe con los resultados.
        
        Args:
            fechas: Lista de fechas en formato dd-mm-yy
            quadrant: Cuadrante de la imagen (por defecto h08v07)
            show_plots: Si mostrar las visualizaciones de matplotlib
            factor_escala: Factor de escala para aumentar la resolución de la imagen (por defecto usa el del constructor)
            
        Returns:
            DataFrame con las mediciones de todas las fechas
        """
        results = []
        
        # Usar el factor de escala pasado como parámetro o el del constructor
        escala_a_usar = factor_escala if factor_escala is not None else self.factor_escala
        
        for fecha in fechas:
            print(f"Procesando fecha: {fecha}")
            try:
                datos = self.get_measures(fecha, quadrant, show_plots=show_plots, factor_escala=escala_a_usar)
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

    def recortar_imagen_solo(self, date_str: str, quadrant: str, factor_escala: int = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Solo recorta la imagen sin hacer mediciones completas.
        
        Args:
            date_str: Fecha en formato dd-mm-yy
            quadrant: Cuadrante de la imagen
            factor_escala: Factor de escala para aumentar la resolución de la imagen (por defecto usa el del constructor)
            
        Returns:
            Tuple con (imagen_recortada, copia_imagen, nuevos_x, nuevos_y)
        """
        year, day, date_obj = parse_date(date_str)
        
        # Usar el factor de escala pasado como parámetro o el del constructor
        escala_a_usar = factor_escala if factor_escala is not None else self.factor_escala

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
                imagen_recortada, nuevos_x, nuevos_y = recortar_imagen(
                    image_matrix, coordenadas_municipio, left_coord, escala_a_usar
                )
                
                if imagen_recortada.size == 0:
                    print("La imagen recortada está vacía. Verifica las coordenadas del municipio.")
                    return None
                    
                copia_imagen = np.clip(imagen_recortada, 0, np.percentile(imagen_recortada, 99))
                
                return imagen_recortada, copia_imagen, nuevos_x, nuevos_y
                
            except Exception as e:
                print(f"Error durante el recorte de la imagen: {e}")
                return None