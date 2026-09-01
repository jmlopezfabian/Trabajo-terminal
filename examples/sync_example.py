from ntl.geometria import SatelliteProcessor


def main() -> None:
    municipio = "Iztapalapa"
    fechas = ["01-01-24"]
    tile = "h08v07"

    processor = SatelliteProcessor(municipio)
    df = processor.run(fechas, tile, show_plots=False)

    if df.empty:
        print("No se obtuvieron resultados.")
    else:
        print("Resultados obtenidos:")
        print(df.head())


if __name__ == "__main__":
    main()

