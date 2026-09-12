import streamlit as st    # interface grafica
import pandas as pd       # tratamento de dados
from sklearn.linear_model import LinearRegression # o tipo de treinamento do modelo
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# tempo de uso de produto X numero de reclamações
X  =  np.array([
    [1,1],
    [5,4],
    [3,3],
    [4,1],
    [4,1],
    [5,0]
])

# vendas = pd.read_csv()


# 0 -> fica - 1 -> cancela
y =  np.array([0,1,1,0,1,1])
modelo = DecisionTreeClassifier()
modelo.fit(X,y)

uso = st.number_input('Quantidade de vezes que o produto fou utilizado: ', value = 0)
reclamacoes = st.number_input('Reclamações: ', value = 0)


st.header('Analise de cancelamento')

if st.button('analisar cliente'):
   if modelo.predict([[uso,reclamacoes]]) == 0:
     
      st.write('CLIENTE CONSOLIDADO')
   else:  
      st.write('POSSIVEL CANCELAMENTO')