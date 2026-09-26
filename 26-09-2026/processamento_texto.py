import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Baixa os recursos necessários do NLTK.
# O parâmetro quiet=True evita mensagens desnecessárias no terminal.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


def normalizar_texto(texto):
    """
    Converte o texto para letras minúsculas
    e remove pontuações.
    """

    # Converte todas as letras para minúsculas.
    texto = texto.lower()

    # Remove pontuações e mantém apenas letras, números e espaços.
    texto = re.sub(r"[^\w\s]", "", texto)

    return texto


def tokenizar_texto(texto):
    """
    Divide o texto em palavras individuais.
    """

    # Primeiro normalizamos o texto.
    texto = normalizar_texto(texto)

    # Depois transformamos o texto em uma lista de palavras.
    tokens = word_tokenize(texto, language="portuguese")

    return tokens


def remover_stopwords(tokens):
    """
    Remove palavras muito comuns da língua portuguesa.
    """

    # Carrega a lista de stopwords em português.
    stopwords_pt = set(stopwords.words("portuguese"))

    # Mantém somente as palavras que NÃO estão
    # na lista de stopwords.
    palavras_relevantes = [
        palavra
        for palavra in tokens
        if palavra not in stopwords_pt
    ]

    return palavras_relevantes