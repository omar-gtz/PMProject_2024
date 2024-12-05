import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configuración inicial de Streamlit
st.title("Estadísticas de los Activos Financieros Seleccionados")

# Definir los ETFs seleccionados
etfs = ["SPY", "EEM", "AGG", "EMB", "GLD"]
start_date = "2010-01-01"
end_date = "2023-12-31"

# Descargar datos de los ETFs
data = {}
for etf in etfs:
    data[etf] = yf.download(etf, start=start_date, end=end_date)["Adj Close"]

# Convertir a DataFrame
df = pd.DataFrame(data)

# Calcular rendimientos diarios
daily_returns = df.pct_change().dropna()

# Calcular estadísticas básicas
stats = pd.DataFrame({
    "Media": daily_returns.mean(),
    "Sesgo": daily_returns.skew(),
    "Curtosis": daily_returns.kurt(),
    "Volatilidad": daily_returns.std(),
    "Sharpe Ratio": daily_returns.mean() / daily_returns.std(),
})

# Calcular VaR y CVaR
confidence_level = 0.95
stats["VaR"] = daily_returns.quantile(1 - confidence_level)
stats["CVaR"] = daily_returns[daily_returns < daily_returns.quantile(1 - confidence_level)].mean()

# Calcular máximo Drawdown
def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()
stats["Max Drawdown"] = daily_returns.apply(max_drawdown)

# Visualizar estadísticas en Streamlit
st.subheader("Estadísticas Calculadas")
st.dataframe(stats)

# Visualizar rendimientos históricos
st.subheader("Rendimientos Históricos")
fig, ax = plt.subplots()
df.plot(ax=ax, title="Precios Históricos de los ETFs")
st.pyplot(fig)

# Visualizar métricas como un gráfico de barras
st.subheader("Métricas de Estadísticas")
fig, ax = plt.subplots(figsize=(10, 6))
stats[["Media", "Volatilidad", "Sharpe Ratio"]].plot(kind="bar", ax=ax)
ax.set_title("Métricas de Estadísticas por ETF")
st.pyplot(fig)
