#Diagnóstico y validación de residuales del modelo ARIMA seleccionado (el ARIMA 0 1 1)
#Importación de librerías y carga de datos
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Definimos rutas relativas
ruta_base = Path(__file__).resolve().parent.parent
ruta_datos = ruta_base / "datos" / "TRM_Mensual_2014_2024.csv"
ruta_resultados = ruta_base / "resultados"
ruta_resultados.mkdir(parents=True, exist_ok=True)

# Se carga la serie y configura la frecuencia de mes a mes
df = pd.read_csv(ruta_datos, sep=";", decimal=",")
df["Fecha"] = pd.to_datetime(df["Año_Mes"])
df.set_index("Fecha", inplace=True)
df.index.freq = "MS"
serie_trm = df["TRM_Promedio_Mensual"]

#Ajuste del modelo ganador: ARIMA(0,1,1) con constante
modelo_ganador = SARIMAX(serie_trm, order=(0, 1, 1), trend="c")
fit_ganador = modelo_ganador.fit(disp=False)

# Extraemos los residuales del modelo
residuos = fit_ganador.resid
print("=== DIAGNÓSTICO Y VALIDACIÓN DE RESIDUALES ===")
print(f"Media de los residuos: {residuos.mean():.4f} (Cercana a 0)")

#Prueba formal de Ljung-Box (para el RB)
# Evaluamos a 6 y 12 rezagos (memoria semestral y anual)
tabla_ljung = acorr_ljungbox(residuos, lags=[6, 12], return_df=True)
print("\n--- PRUEBA DE LJUNG-BOX (AUTOCORRELACIÓN RESIDUAL) ---")
print(tabla_ljung)

if (tabla_ljung["lb_pvalue"] > 0.05).all():
	print(
		"\nConclusión Ljung-Box: Los residuos SON RUIDO BLANCO "
		"(p-valores > 0.05). No hay autocorrelación residual."
	)
else:
	print("\nConclusión Ljung-Box: Los residuos presentan autocorrelación significativa.")

# Prueba formal de normalidad de Jarque-Bera
jb_stat, jb_pvalue = stats.jarque_bera(residuos)
skew = stats.skew(residuos)
kurt = stats.kurtosis(residuos)

print("\n--- PRUEBA DE JARQUE-BERA (NORMALIDAD DE RESIDUOS) ---")
print(f"Estadístico Jarque-Bera: {jb_stat:.4f}")
print(f"p-valor: {jb_pvalue:.4f}")
print(f"Asimetría (Skewness): {skew:.4f}")
print(f"Curtosis: {kurt:.4f}")

if jb_pvalue > 0.05:
	print("Conclusión Jarque-Bera: Los residuos SIGUEN una distribución normal (p-valor > 0.05).")
else:
	print(
		"Conclusión Jarque-Bera: Los residuos presentan desviaciones de normalidad "
		"(común en series financieras por colas pesadas)."
	)

# Gráficas de diagnóstico de residuos
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Gráfica temporal de residuos
axes[0, 0].plot(residuos, color="purple", linewidth=1)
axes[0, 0].axhline(0, color="red", linestyle="--")
axes[0, 0].set_title("Residuos a lo largo del tiempo", fontsize=11)
axes[0, 0].grid(True, linestyle="--", alpha=0.5)

# 2. Histograma de residuos frente a distribución normal
axes[0, 1].hist(
	residuos, bins=20, density=True, alpha=0.6, color="teal", edgecolor="black"
)
mu, std = residuos.mean(), residuos.std()
x_vals = np.linspace(residuos.min(), residuos.max(), 100)
axes[0, 1].plot(
	x_vals,
	stats.norm.pdf(x_vals, mu, std),
	"r-",
	linewidth=2,
	label="Normal Teórica",
)
axes[0, 1].set_title("Histograma de Residuos", fontsize=11)
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle="--", alpha=0.5)

# 3. Correlograma ACF de residuos
plot_acf(residuos, ax=axes[1, 0], lags=20, alpha=0.05)
axes[1, 0].set_title("ACF de Residuos (Ruido Blanco)", fontsize=11)
axes[1, 0].grid(True, linestyle="--", alpha=0.5)

# 4. Gráfico Q-Q (normalidad)
stats.probplot(residuos, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Gráfico Q-Q Plot de Residuos", fontsize=11)
axes[1, 1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
ruta_grafica = ruta_resultados / "05_diagnostico_residuos.png"
plt.savefig(ruta_grafica)
plt.show()
print(f"\n¡ÉXITO de las pruebas! va para: {ruta_grafica}")
