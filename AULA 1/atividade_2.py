import streamlit as st

st.header('Entrada de dados')


nome = st.text_input('Digite seu nome: ')
idade = st.text_input('Digite sua idade: ')
aceitacao =st.checkbox(label=('Aceito os termos de uso'), value=False, key=None, help=None,
 on_change=None, args=None, kwargs=None, disabled=False,
  label_visibility="visible", width="content", wrap=None,
   bind=None, persist_state=None)


if aceitacao: 
    st.success('Termo aceito!')
    if st.button('Exibir'):
        st.info(nome) 
        st.info(idade) 
elif not aceitacao:
    st.warning('Termo não aceito!')

