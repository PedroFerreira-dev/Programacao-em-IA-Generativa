import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

dados = pd.read_csv('vendas.csv')
st.write(dados)

#Organizar os dados
x = dados[['mes']]
y = dados['vendas']

# Treinar o modelo
model = LinearRegression()
model.fit(x, y)
mes = 9

df = model.predict(pd.DataFrame({'mes': [mes]}))[0]

st.write(f'Previsao do mes: R${df:.2f}')
