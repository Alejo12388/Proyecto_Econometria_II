from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


#Importación de librerías y definición de rutas relativas
ruta_base = Path(__file__).resolve().parent.parent
ruta_datos = ruta_base / "datos" / "TRM_Mensual_2014_2024.csv"
ruta_resultados = ruta_base / "resultados"
ruta_resultados.mkdir(parents=True, exist_ok=True)

print("Ruta de datos cargada correctamente:", ruta_datos)

#Carga y preparación de la serie de tiempo
df = pd.read_csv(ruta_datos, sep=";", decimal=",")
df["Fecha"] = pd.to_datetime(df["Año_Mes"])
df.set_index("Fecha", inplace=True)

serie_trm = df["TRM_Promedio_Mensual"]

print("\n--- RESUMEN DE LA SERIE ---")
print(f"Total de observaciones (T): {len(serie_trm)}")
print(serie_trm.describe())

#Gráfica de la serie en niveles y prueba ADF
plt.figure(figsize=(10, 4))
plt.plot(
	serie_trm,
	color="navy",
	linewidth=1.8,
	label="TRM Mensual (COP/USD)",
)
plt.title("Tasa Representativa del Mercado en Niveles (2014-2024)", fontsize=12)
plt.xlabel("Fecha")
plt.ylabel("Pesos Colombianos (COP)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(ruta_resultados / "01_trm_niveles.png")
plt.show()

res_adf_niveles = adfuller(serie_trm)
print("\n--- PRUEBA DICKEY-FULLER AUMENTADA (NIVELES) ---")
print(f"Estadístico ADF: {res_adf_niveles[0]:.4f}")
print(f"p-valor: {res_adf_niveles[1]:.4f}")
if res_adf_niveles[1] < 0.05:
	print("Conclusión: La serie ES estacionaria en niveles (p-valor < 0.05).")
else:
	print(
		"Conclusión: La serie NO es estacionaria en niveles "
		"(p-valor >= 0.05). Requiere diferenciación."
	)

#Primera diferencia y verificación de estacionariedad
serie_diff = serie_trm.diff().dropna()

plt.figure(figsize=(10, 4))
plt.plot(
	serie_diff,
	color="darkgreen",
	linewidth=1.5,
	label="Primera Diferencia de la TRM (Delta TRM)",
)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.title("Primera Diferencia de la TRM (Variación Mensual en COP)", fontsize=12)
plt.xlabel("Fecha")
plt.ylabel("Cambio Mensual (COP)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(ruta_resultados / "02_trm_diferenciada.png")
plt.show()

res_adf_diff = adfuller(serie_diff)
print("\n--- PRUEBA DICKEY-FULLER AUMENTADA (SERIE DIFERENCIADA) ---")
print(f"Estadístico ADF: {res_adf_diff[0]:.4f}")
print(f"p-valor: {res_adf_diff[1]:.4f}")
if res_adf_diff[1] < 0.05:
	print("Conclusión: La serie diferenciada ES ESTACIONARIA. Determinamos que d = 1.")
else:
	print("Conclusión: Requiere una segunda diferencia (d = 2).")

#Identificación del modelo: gráficas ACF y PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(serie_diff, ax=axes[0], lags=24, alpha=0.05)
axes[0].set_title("Función de Autocorrelación (ACF) - Delta TRM", fontsize=11)
axes[0].grid(True, linestyle="--", alpha=0.5)

plot_pacf(serie_diff, ax=axes[1], lags=24, alpha=0.05, method="ywm")
axes[1].set_title(
	"Función de Autocorrelación Parcial (PACF) - Delta TRM", fontsize=11
)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(ruta_resultados / "03_acf_pacf.png")
plt.show()
