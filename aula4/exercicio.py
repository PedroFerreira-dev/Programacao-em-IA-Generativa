import pandas as pd
import tensorflow as tf


# ============================================================
# 1. CRIANDO OS DADOS
# ============================================================

estudos = pd.DataFrame({
    'notas': [1, 2, 4, 6, 8, 10],
    'horas': [2, 4, 5, 7, 9, 10]
})


# ============================================================
# 2. SEPARANDO ENTRADA E SAÍDA
# ============================================================

# Entrada do modelo: horas de estudo
X = estudos[['horas']]

# Saída esperada: notas
y = estudos['notas']


# ============================================================
# 3. CRIANDO O MODELO
# ============================================================

modelo = tf.keras.Sequential([
    
    # Define que o modelo recebe uma única informação:
    # a quantidade de horas estudadas.
    tf.keras.Input(shape=(1,)),
    
    # Cria um neurônio para fazer a previsão.
    tf.keras.layers.Dense(
        1,
        activation='linear'
    )
])


# ============================================================
# 4. COMPILANDO O MODELO
# ============================================================

modelo.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)


# ============================================================
# 5. TREINANDO O MODELO
# ============================================================

modelo.fit(
    X,
    y,
    epochs=1000,
    verbose=0
)


# ============================================================
# 6. CRIANDO UM NOVO DADO
# ============================================================

# Queremos prever a nota de um aluno
# que estudou 6 horas.

horas_estudo = pd.DataFrame({
    'horas': [6]
})


# ============================================================
# 7. FAZENDO A PREVISÃO
# ============================================================

previsao = modelo.predict(
    horas_estudo,
    verbose=0
)


# ============================================================
# 8. MOSTRANDO O RESULTADO
# ============================================================

print(
    f"Nota prevista para 6 horas de estudo: {previsao[0][0]:.2f}"
)

