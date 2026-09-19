# ============================================================
# ANÁLISE DE CRÉDITO PARA FINANCIAMENTO DE MOTO
# Streamlit + TensorFlow
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="MotoCred",
    page_icon="🏍️",
    layout="centered"
)


# ============================================================
# 2. TÍTULO DA APLICAÇÃO
# ============================================================

st.title("🏍️ MotoCred")
st.subheader("Análise inteligente de crédito")

st.write(
    "Preencha seus dados abaixo para realizar uma simulação "
    "de financiamento da sua primeira moto."
)

st.info(
    "Esta aplicação é uma simulação educacional. "
    "A análise não representa uma decisão bancária real."
)


# ============================================================
# 3. CRIANDO UMA BASE DE DADOS SINTÉTICA
# ============================================================
#
# Como o projeto não possui uma base histórica real, criamos
# exemplos para que o TensorFlow possa aprender uma relação
# entre:
#
# - idade
# - renda
# - valor da parcela
#
# e o resultado:
#
# 0 = crédito negado
# 1 = crédito aprovado
#
# ATENÇÃO:
# Esses dados são fictícios e servem apenas para demonstrar
# o funcionamento do modelo.
# ============================================================

dados = pd.DataFrame({
    "idade": [
        18, 19, 20, 22, 25, 28, 30, 32, 35, 40,
        21, 24, 27, 31, 36, 42, 45, 23, 29, 34
    ],

    "renda": [
        1500, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000,
        2200, 2800, 3200, 3800, 4500, 5500, 6500, 2400, 3300, 4200
    ],

    "parcela": [
        900, 1000, 1100, 900, 1000, 1100, 1200, 1300, 1400, 1500,
        700, 800, 900, 1000, 1100, 1200, 1300, 1000, 1200, 1400
    ],

    "aprovado": [
        0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 0, 1, 1
    ]
})


# ============================================================
# 4. SEPARANDO ENTRADAS E SAÍDA
# ============================================================

X = dados[
    ["idade", "renda", "parcela"]
]

y = dados["aprovado"]


# ============================================================
# 5. NORMALIZAÇÃO DOS DADOS
# ============================================================
#
# Idade, renda e parcela possuem escalas muito diferentes.
#
# Exemplo:
#
# idade  = 25
# renda  = 3000
# parcela = 1000
#
# Para facilitar o aprendizado da rede, normalizamos os dados.
# ============================================================

media = X.mean()
desvio = X.std()

X_normalizado = (X - media) / desvio


# ============================================================
# 6. CRIANDO O MODELO
# ============================================================
#
# Temos 3 entradas:
#
# 1. idade
# 2. renda
# 3. parcela
#
# A saída será uma probabilidade entre 0 e 1.
#
# Sigmoid é utilizada porque estamos tratando de uma
# classificação binária:
#
# 0 = negado
# 1 = aprovado
# ============================================================

modelo = tf.keras.Sequential([

    tf.keras.Input(shape=(3,)),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        8,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 7. COMPILANDO O MODELO
# ============================================================

modelo.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 8. TREINANDO O MODELO
# ============================================================

modelo.fit(
    X_normalizado,
    y,
    epochs=300,
    verbose=0
)


# ============================================================
# 9. FORMULÁRIO
# ============================================================

st.divider()

st.header("📋 Simule seu financiamento")

with st.form("formulario_credito"):

    idade = st.number_input(
        "Idade",
        min_value=18,
        max_value=100,
        value=25,
        step=1
    )

    renda = st.number_input(
        "Renda mensal",
        min_value=0.0,
        value=3000.0,
        step=100.0,
        format="%.2f"
    )

    parcela = st.number_input(
        "Valor da parcela",
        min_value=0.0,
        value=900.0,
        step=50.0,
        format="%.2f"
    )

    enviar = st.form_submit_button(
        "🔎 Analisar crédito"
    )


# ============================================================
# 10. REALIZANDO A ANÁLISE
# ============================================================

if enviar:

    # --------------------------------------------------------
    # Validação básica
    # --------------------------------------------------------

    if renda <= 0:
        st.error("Informe uma renda mensal válida.")

    elif parcela <= 0:
        st.error("Informe um valor de parcela válido.")

    else:

        # ----------------------------------------------------
        # Calculando o comprometimento da renda
        # ----------------------------------------------------

        comprometimento = parcela / renda

        # ----------------------------------------------------
        # Criando os dados do novo cliente
        # ----------------------------------------------------

        novo_cliente = pd.DataFrame({
            "idade": [idade],
            "renda": [renda],
            "parcela": [parcela]
        })

        # ----------------------------------------------------
        # Aplicando a mesma normalização utilizada
        # no treinamento
        # ----------------------------------------------------

        novo_cliente_normalizado = (
            novo_cliente - media
        ) / desvio

        # ----------------------------------------------------
        # Fazendo a previsão
        # ----------------------------------------------------

        probabilidade = modelo.predict(
            novo_cliente_normalizado,
            verbose=0
        )[0][0]

        # ----------------------------------------------------
        # Convertendo para porcentagem
        # ----------------------------------------------------

        percentual = probabilidade * 100


        # ====================================================
        # 11. RESULTADO
        # ====================================================

        st.divider()

        st.header("📊 Resultado da análise")

        if probabilidade >= 0.5:

            st.success(
                "✅ Crédito aprovado na simulação!"
            )

            st.metric(
                "Probabilidade estimada",
                f"{percentual:.1f}%"
            )

            st.write(
                "Com os dados informados, o modelo classificou "
                "a solicitação como aprovada."
            )

        else:

            st.error(
                "❌ Crédito não aprovado na simulação."
            )

            st.metric(
                "Probabilidade estimada",
                f"{percentual:.1f}%"
            )

            # =================================================
            # 12. SUGESTÃO DE PARCELA
            # =================================================
            #
            # Se a parcela compromete uma porcentagem muito
            # grande da renda, sugerimos reduzir o valor.
            #
            # Aqui usamos 30% apenas como uma regra didática
            # para o projeto.
            # =================================================

            parcela_sugerida = renda * 0.30

            st.warning(
                "💡 Uma alternativa é reduzir o valor da parcela."
            )

            st.write(
                f"Parcela atual: R$ {parcela:,.2f}"
            )

            st.write(
                f"Parcela sugerida: R$ {parcela_sugerida:,.2f}"
            )

            st.write(
                "Essa sugestão é apenas uma simulação baseada "
                "no comprometimento de renda definido neste projeto."
            )


        # ====================================================
        # 13. INDICADOR DE COMPROMETIMENTO DA RENDA
        # ====================================================

        st.divider()

        st.subheader("💰 Comprometimento da renda")

        st.progress(
            min(comprometimento, 1.0)
        )

        st.write(
            f"Você comprometeria aproximadamente "
            f"**{comprometimento * 100:.1f}%** da sua renda "
            f"com essa parcela."
        )


# ============================================================
# 14. RODAPÉ
# ============================================================

st.divider()

st.caption(
    "MotoCred • Simulador educacional desenvolvido com "
    "Python, Streamlit, Pandas e TensorFlow."
)

