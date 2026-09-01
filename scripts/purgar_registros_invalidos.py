"""
Detecta y purga los registros contaminados por valores de relleno.

Hasta que se enmascaró ``_FillValue``, un cuadrante sin medición se sumaba como
si 65535 fuera radianza. En el histórico hay cuatro fechas así, y sus registros
salieron con sumas 229 veces mayores que las normales: una media de 65535 exactos
y un máximo idéntico, porque *todos* los píxeles eran relleno.

Un registro se marca como contaminado cuando su máximo alcanza el valor de
relleno. Eso cubre tanto el caso total —el cuadrante entero sin dato— como el
parcial, en el que solo algunos píxeles lo estaban y la media queda por debajo;
el parcial es más traicionero porque el resto de las cifras parece plausible.

No escribe sobre el archivo de entrada. El histórico vive en un espejo de Drive
compartido con otras personas, y sobrescribirlo ahí propagaría el cambio sin que
nadie lo haya revisado.

Uso:
    python scripts/purgar_registros_invalidos.py ENTRADA [--salida ruta] [--solo-reportar]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Valor de relleno del producto, en cuentas digitales. Las series anteriores al
# cambio están sin escalar; si alguna trae ya el factor aplicado, el relleno
# aparece como 6553.5.
RELLENO_DN = 65535.0
RELLENO_ESCALADO = 6553.5
TOLERANCIA = 0.5

COLUMNA_MAXIMO = "Maximo_de_radianza"
COLUMNA_MEDIA = "Media_de_radianza"


def _leer(ruta: str) -> pd.DataFrame:
    if ruta.endswith(".parquet"):
        return pd.read_parquet(ruta)
    return pd.read_csv(ruta, parse_dates=["Fecha"])


def _escribir(df: pd.DataFrame, ruta: str) -> None:
    if ruta.endswith(".parquet"):
        df.to_parquet(ruta, index=False)
    else:
        df.to_csv(ruta, index=False)


def marcar_contaminados(df: pd.DataFrame) -> pd.Series:
    """Registros cuyo máximo coincide con el valor de relleno."""
    maximo = df[COLUMNA_MAXIMO]
    # Los paréntesis no son opcionales: `|` liga más fuerte que `<`.
    return (((maximo - RELLENO_DN).abs() < TOLERANCIA)
            | ((maximo - RELLENO_ESCALADO).abs() < TOLERANCIA))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", help="CSV o Parquet con las mediciones")
    parser.add_argument("--salida", help="Ruta de escritura (por defecto, ENTRADA_limpio.ext)")
    parser.add_argument("--solo-reportar", action="store_true",
                        help="Enumera lo contaminado sin escribir nada")
    args = parser.parse_args()

    df = _leer(args.entrada)
    if COLUMNA_MAXIMO not in df.columns:
        print(f"ERROR: {args.entrada} no tiene la columna {COLUMNA_MAXIMO}")
        return 1

    malos = marcar_contaminados(df)
    print(f"registros: {len(df):,}   contaminados: {int(malos.sum()):,}")

    if not malos.any():
        print("Nada que purgar.")
        return 0

    sucios = df[malos]
    print(f"\n{'fecha':12s} {'municipios':>11s} {'totalmente relleno':>20s} {'parcial':>9s}")
    for fecha, grupo in sucios.groupby(sucios["Fecha"].astype(str)):
        parcial = int((grupo[COLUMNA_MEDIA] < RELLENO_DN - TOLERANCIA).sum())
        print(f"{fecha:12s} {len(grupo):11d} {len(grupo)-parcial:20d} {parcial:9d}")

    sanos = df[~malos]
    if COLUMNA_MEDIA in df.columns and len(sanos):
        print(f"\nsuma mediana de un registro sano:        {sanos['Suma_de_radianza'].median():>14,.0f}")
        print(f"suma mediana de un registro contaminado: {sucios['Suma_de_radianza'].median():>14,.0f}")

    if args.solo_reportar:
        print("\n--solo-reportar: no se escribió nada.")
        return 0

    salida = args.salida
    if not salida:
        raiz, ext = os.path.splitext(args.entrada)
        salida = f"{raiz}_limpio{ext}"
    if os.path.abspath(salida) == os.path.abspath(args.entrada):
        print("ERROR: la salida no puede ser el archivo de entrada")
        return 1

    _escribir(sanos, salida)
    print(f"\nEscrito: {salida} ({len(sanos):,} registros, {int(malos.sum()):,} purgados)")
    print(f"Sin tocar: {args.entrada}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
