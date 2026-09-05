import streamlit as st

st.header('Seletor de cursos: ')

curso = st.selectbox('Seleione o curso que deseja: ', ['Python', 'Java', 'C#', 'JavaScript', 'PHP'])
tecnologias = st.multiselect('Escolhas as tecnologias', ['HTML', 'CSS', 'SQL', 'Git'] ) 

tecnologias_formatado = ', '.join(tecnologias)

st.write('Curso selecionado: ', curso )
st.write('Tecnologias selecionadas: ', tecnologias_formatado)