from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


#Importación de librerías y carga de datos
ruta_base = Path(__file__).resolve().parent.parent
ruta_datos = ruta_base / "datos" / "TRM_Mensual_2014_2024.csv"
ruta_resultados = ruta_base / "resultados"
ruta_resultados.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ruta_datos, sep=";", decimal=",")
df["Fecha"] = pd.to_datetime(df["Año_Mes"])
df.set_index("Fecha", inplace=True)
serie_trm = df["TRM_Promedio_Mensual"]


#Estimación manual de la matriz de modelos candidatos
candidatos = [
	((0, 1, 0), "c", "ARIMA(0,1,0) con constante (Random Walk with Drift)"),
	((0, 1, 0), "n", "ARIMA(0,1,0) sin constante (Random Walk puro)"),
	((1, 1, 0), "c", "ARIMA(1,1,0) con constante"),
	((1, 1, 0), "n", "ARIMA(1,1,0) sin constante"),
	((0, 1, 1), "c", "ARIMA(0,1,1) con constante"),
	((0, 1, 1), "n", "ARIMA(0,1,1) sin constante"),
	((1, 1, 1), "c", "ARIMA(1,1,1) con constante"),
	((1, 1, 1), "n", "ARIMA(1,1,1) sin constante"),
]

resultados_tabla = []
print("=== ESTIMACIÓN Y COMPARACIÓN DE MODELOS ARIMA ===")
for order, trend, nombre in candidatos:
	modelo = SARIMAX(serie_trm, order=order, trend=trend)
	ajuste = modelo.fit(disp=False)
	resultados_tabla.append(
		{
			"Modelo": nombre,
			"Order (p,d,q)": order,
			"Constante": "Sí" if trend == "c" else "No",
			"AIC": round(ajuste.aic, 2),
			"BIC": round(ajuste.bic, 2),
			"Log-Likelihood": round(ajuste.llf, 2),
		}
	)

df_comparacion = pd.DataFrame(resultados_tabla)
df_comparacion.sort_values(by="AIC", inplace=True)
df_comparacion.reset_index(drop=True, inplace=True)
print("\n--- TABLA COMPARATIVA DE MODELOS (ORDENADOS POR MENOR AIC) ---")
print(df_comparacion.to_string(index=False))

archivo_comparacion = ruta_resultados / "04_comparacion_modelos_aic.csv"
df_comparacion.to_csv(archivo_comparacion, index=False)
print(f"\n¡Tabla comparativa guardada en: {archivo_comparacion}!")


#Ajuste detallado del modelo ganador
mejor_orden = df_comparacion.loc[0, "Order (p,d,q)"]
mejor_trend = "c" if df_comparacion.loc[0, "Constante"] == "Sí" else "n"
mejor_nombre = df_comparacion.loc[0, "Modelo"]
print("\n==================================================")
print(f"SELECCIONADO COMO MEJOR MODELO: {mejor_nombre}")
print("==================================================")

modelo_optimo = SARIMAX(serie_trm, order=mejor_orden, trend=mejor_trend)
fit_optimo = modelo_optimo.fit(disp=False)
print(fit_optimo.summary())
