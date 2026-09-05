import streamlit as st
import pandas as pd

# Carrega os dados do CSV
dados = pd.read_csv('vendas.csv')

st.header('Calculadora STREAMLIT')
st.write('Adicione os números para calcular')

n1 = st.number_input('Digite um numero 1: ', min_value=0)
n2 = st.number_input('Digite um numero 2: ', min_value=0)

soma_, sub_, multi_, div_ = st.columns(4)

# Bloco da Calculadora
if soma_.button('+'):
    soma = n1 + n2
    st.info(soma) 
elif sub_.button('-'):
    sub = n1 - n2
    st.info(sub)
elif multi_.button('*'):
    multi = n1 * n2
    st.info(multi)
elif div_.button(':'):
    # Correção: Evita erro de divisão por zero caso n2 seja 0
    if n2 != 0:
        div = n1 / n2
        st.info(div)
    else:
        st.error("Não é possível dividir por zero.")


if st.button('mostrar mapa'):
    x = st.map()


st.header('Análise de dados')
st.table(dados)
st.bar_chart(dados, x='ano', y='lucro')
st.scatter_chart(dados, x='venda', y='lucro')
st.line_chart(dados, x='ano', y='venda')  
