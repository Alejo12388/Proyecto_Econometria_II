from pathlib import Path

import pandas as pd

# 1. Definir la ruta relativa usando pathlib (como exige la guía)
ruta_base = Path(__file__).resolve().parent.parent
ruta_datos = ruta_base / "datos" / "TRM_Mensual_2014_2024.csv"

# 2. Cargar el archivo CSV
df = pd.read_csv(ruta_datos, sep=";", decimal=",")

# 3. Mostrar las primeras 5 filas en pantalla
print("--- DATOS CARGADOS CON ÉXITO USANDO PATHLIB ---")
print(df.head())
