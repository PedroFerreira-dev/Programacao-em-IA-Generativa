import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.header('PREVISÃO DE VENDAS')

# Dados de vendas
dados_vendas = pd.DataFrame({
    'investimento': [100, 200, 300, 400, 500],
    'faturamento': [150, 250, 350, 450, 550]
})

# Mostrar os dados
st.write(dados_vendas)

# Separar os dados para treinamento
x = dados_vendas[['investimento']]
y = dados_vendas['faturamento']

# Treinar o modelo
model = LinearRegression()
model.fit(x, y)

# Entrada do usuário
investimento = st.number_input(
    'Digite o investimento',
    value=150
)

# Fazer previsão
if st.button('Analisar'):
    previsao = model.predict(
        pd.DataFrame({'investimento': [investimento]})
    )[0]

    st.write(f'Faturamento previsto: R$ {previsao:.2f}')
