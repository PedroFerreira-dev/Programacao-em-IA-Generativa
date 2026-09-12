import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

dados = pd.read_csv('vendas.csv')

df = pd.DataFrame(dados)

print(df)
