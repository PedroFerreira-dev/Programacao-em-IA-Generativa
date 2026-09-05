import streamlit as st
import pandas as pd

# Carrega os dados do CSV
dados = pd.read_csv('dados.csv')

st.header('Tabela de informações de funcionarios')
st.write('Exibindo:')


st.header('Análise de dados')
st.table(dados)
st.dataframe(dados)