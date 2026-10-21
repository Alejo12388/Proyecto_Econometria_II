from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Rutas relativas
ruta_base = Path(__file__).resolve().parent.parent
ruta_datos = ruta_base / "datos" / "TRM_Mensual_2014_2024.csv"
ruta_resultados = ruta_base / "resultados"
ruta_resultados.mkdir(parents=True, exist_ok=True)

# Cargamos la serie histórica y fijamos frecuencia mes a mes, de nuevo
df = pd.read_csv(ruta_datos, sep=";", decimal=",")
df["Fecha"] = pd.to_datetime(df["Año_Mes"])
df.set_index("Fecha", inplace=True)
df.index.freq = "MS"
serie_trm = df["TRM_Promedio_Mensual"]

# se ajusta el modelo y se establece el pronóstico de 10 periodos adelante (h = 10 periodos)
modelo_optimo = SARIMAX(serie_trm, order=(0, 1, 1), trend="c")
fit_optimo = modelo_optimo.fit(disp=False)

# Generamos el pronóstico para 10 meses adelante
pasos = 10
pronostico_obj = fit_optimo.get_forecast(steps=pasos)

# Extraemos valores puntuales e intervalos de confianza del 95%
df_pronostico = pronostico_obj.summary_frame(alpha=0.05)
df_pronostico = df_pronostico[["mean", "mean_ci_lower", "mean_ci_upper"]]
df_pronostico.columns = [
	"TRM_Pronosticada",
	"IC_Inferior_95",
	"IC_Superior_95",
]

print("=== TABLA DE PRONÓSTICOS A 10 MESES (COP/USD) ===")
print(df_pronostico.round(2))

#Ahora se guarda la tabla en la carpeta resultados
archivo_tabla = ruta_resultados / "07_tabla_pronostico.csv"
df_pronostico.to_csv(archivo_tabla)
print(f"\n¡Tabla de pronóstico guardada en: {archivo_tabla}!")

# Gráficamos el pronóstico con intervalos de confianza
plt.figure(figsize=(11, 5))
historico_reciente = serie_trm["2021-01-01":]
plt.plot(
	historico_reciente.index,
	historico_reciente,
	color="navy",
	linewidth=2,
	label="TRM Histórica",
)
plt.plot(
	df_pronostico.index,
	df_pronostico["TRM_Pronosticada"],
	color="red",
	linestyle="--",
	marker="o",
	linewidth=2,
	label="Pronóstico ARIMA(0,1,1) c",
)
plt.fill_between(
	df_pronostico.index,
	df_pronostico["IC_Inferior_95"],
	df_pronostico["IC_Superior_95"],
	color="pink",
	alpha=0.5,
	label="Intervalo de Confianza (95%)",
)

plt.title(
	"Pronóstico de la Tasa Representativa del Mercado (TRM) - Próximos 10 Meses",
	fontsize=12,
)
plt.xlabel("Fecha")
plt.ylabel("Pesos Colombianos (COP)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")
plt.tight_layout()

# Guardamos la imagen final
archivo_grafica = ruta_resultados / "06_pronostico_trm.png"
plt.savefig(archivo_grafica)
plt.show()
print(f"\n¡ÉXITO del modelo! Gráfica queda guardada en: {archivo_grafica}")
