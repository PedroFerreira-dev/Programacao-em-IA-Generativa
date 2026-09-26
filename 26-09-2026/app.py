import tensorflow as tf
import streamlit as st

from processamento_texto import (
    tokenizar_texto,
    remover_stopwords
)

from analise import (
    contar_frequencia,
    identificar_palavras_negativas,
    detectar_palavras_chave,
    classificar_sentimento,
    classificar_setor
)

from modelo import (
    carregar_dados,
    preparar_dados,
    treinar_modelo
)


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Análise de Textos",
    page_icon="📝",
    layout="wide"
)


# ============================================================
# TÍTULO
# ============================================================

st.title("📝 Sistema de Análise de Textos")

st.write(
    """
    Sistema simples para análise de mensagens de clientes
    utilizando NLTK, TensorFlow e Streamlit.
    """
)


# ============================================================
# PREPARAÇÃO DO MODELO
# ============================================================

@st.cache_resource
def preparar_modelo():
    """
    Carrega os dados, prepara os textos e treina
    o modelo de classificação de sentimentos.
    """

    # Carrega os dados do arquivo CSV.
    dados = carregar_dados()

    # Separa os textos das classes.
    textos, classes = preparar_dados(dados)

    # Cria e treina o modelo.
    modelo = treinar_modelo(
        textos,
        classes
    )

    return modelo


# Prepara o modelo.
modelo = preparar_modelo()


# ============================================================
# CAMPO PARA DIGITAR O TEXTO
# ============================================================

texto = st.text_area(
    "Digite uma mensagem:",
    placeholder=(
        "Exemplo: Meu pagamento apresentou erro "
        "e quero cancelar a compra."
    )
)


# ============================================================
# BOTÃO DE ANÁLISE
# ============================================================

if st.button("Analisar texto"):

    # Verifica se o usuário digitou alguma coisa.
    if not texto.strip():

        st.warning(
            "Digite uma mensagem para realizar a análise."
        )

    else:

        # ====================================================
        # 1. TOKENIZAÇÃO
        # ====================================================

        tokens = tokenizar_texto(texto)

        st.subheader("1. Tokenização")

        st.write(tokens)


        # ====================================================
        # 2. REMOÇÃO DE STOPWORDS
        # ====================================================

        palavras_relevantes = remover_stopwords(tokens)

        st.subheader("2. Remoção de stopwords")

        st.write(palavras_relevantes)


        # ====================================================
        # 3. FREQUÊNCIA DAS PALAVRAS
        # ====================================================

        frequencia = contar_frequencia(
            palavras_relevantes
        )

        st.subheader("3. Frequência das palavras")

        st.write(
            frequencia.most_common()
        )


        # ====================================================
        # 4. PALAVRAS NEGATIVAS
        # ====================================================

        negativas = identificar_palavras_negativas(
            palavras_relevantes
        )

        st.subheader("4. Palavras negativas")

        if negativas:

            st.error(
                f"Palavras negativas encontradas: "
                f"{negativas}"
            )

        else:

            st.success(
                "Nenhuma palavra negativa encontrada."
            )


        # ====================================================
        # 5. SENTIMENTO POR REGRAS
        # ====================================================

        sentimento = classificar_sentimento(
            palavras_relevantes
        )

        st.subheader("5. Sentimento")

        st.info(
            f"Classificação: {sentimento}"
        )


        # ====================================================
        # 6. PALAVRAS-CHAVE
        # ====================================================

        palavras_chave = detectar_palavras_chave(
            palavras_relevantes
        )

        st.subheader("6. Palavras-chave")

        if palavras_chave:

            st.write(palavras_chave)

        else:

            st.write(
                "Nenhuma palavra-chave encontrada."
            )


        # ====================================================
        # 7. PALAVRAS MAIS FREQUENTES
        # ====================================================

        palavras_frequentes = frequencia.most_common(5)

        st.subheader("7. Palavras mais frequentes")

        if palavras_frequentes:

            for palavra, quantidade in palavras_frequentes:

                st.write(
                    f"**{palavra}** → "
                    f"{quantidade} ocorrência(s)"
                )

        else:

            st.write(
                "Nenhuma palavra encontrada."
            )


        # ====================================================
        # 8. CLASSIFICAÇÃO DO SETOR
        # ====================================================

        setor = classificar_setor(
            palavras_relevantes
        )

        st.subheader("8. Setor")

        st.info(
            f"Setor identificado: {setor}"
        )


        # ====================================================
        # 9. TEXTO NORMALIZADO
        # ====================================================

        texto_normalizado = " ".join(
            palavras_relevantes
        )

        st.subheader("9. Texto normalizado")

        st.code(
            texto_normalizado
        )


        # ====================================================
        # 10. ANÁLISE COM TENSORFLOW
        # ====================================================

        st.subheader(
            "10. Análise com TensorFlow"
        )

        # Transformamos o texto em um Tensor do TensorFlow.
        #
        # [texto] significa que estamos criando uma
        # entrada contendo um único texto.
        #
        # dtype=tf.string informa explicitamente que
        # a entrada contém texto.
        entrada = tf.constant(
            [texto],
            dtype=tf.string
        )

        # Enviamos o Tensor diretamente para o modelo.
        #
        # training=False informa que estamos apenas
        # fazendo uma previsão e não treinando o modelo.
        previsao = modelo(
            entrada,
            training=False
        ).numpy()[0][0]

        # O modelo utiliza sigmoid na última camada.
        #
        # O resultado fica entre 0 e 1.
        #
        # >= 0.5 -> Positivo
        # < 0.5  -> Negativo
        if previsao >= 0.5:

            resultado_modelo = "Positivo"

        else:

            resultado_modelo = "Negativo"

        st.write(
            f"Sentimento previsto pelo modelo: "
            f"**{resultado_modelo}**"
        )

        st.write(
            f"Probabilidade positiva: "
            f"**{previsao:.2%}**"
        )
