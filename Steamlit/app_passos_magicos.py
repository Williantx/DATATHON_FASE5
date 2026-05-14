import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================

st.set_page_config(
    page_title='Modelo Preditivo',
    page_icon='📊',
    layout='wide'
)

# =========================
# CARREGAMENTO DO MODELO
# =========================

model = joblib.load('modelo.pkl')

# =========================
# FEATURES DO MODELO
# =========================

features = [
    'IDA',
    'IEG',
    'IAA',
    'IPS',
    'IPP',
    'Fase_Num',
    'IPV',
    'Anos_No_Programa'
]

# =========================
# SIDEBAR
# =========================

st.sidebar.title('Configurações')

threshold = st.sidebar.slider(
    'Threshold de risco',
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.01
)

# =========================
# TÍTULO
# =========================

st.title('📊 Modelo Preditivo de Risco')

st.markdown(
    '''
    Aplicação para análise preditiva baseada no modelo treinado
    utilizando RandomForestClassifier.
    '''
)

# =========================
# FORMULÁRIO
# =========================

st.subheader('Inserir Dados do Aluno')

col1, col2, col3, col4 = st.columns(4)

with col1:
    IDA = st.number_input('IDA', value=7.0)
    IEG = st.number_input('IEG', value=7.0)

with col2:
    IAA = st.number_input('IAA', value=7.0)
    IPS = st.number_input('IPS', value=7.0)

with col3:
    IPP = st.number_input('IPP', value=7.0)
    Fase_Num = st.number_input('Fase_Num', value=3)

with col4:
    IPV = st.number_input('IPV', value=7.0)
    Anos_No_Programa = st.number_input('Anos_No_Programa', value=2)

# =========================
# DATAFRAME DE ENTRADA
# =========================

novo_aluno = pd.DataFrame({
    'IDA': [IDA],
    'IEG': [IEG],
    'IAA': [IAA],
    'IPS': [IPS],
    'IPP': [IPP],
    'Fase_Num': [Fase_Num],
    'IPV': [IPV],
    'Anos_No_Programa': [Anos_No_Programa]
})

# =========================
# PREVISÃO
# =========================

if st.button('Gerar Análise'):

    probabilidade = model.predict_proba(novo_aluno)[0][1]

    predicao = int(probabilidade > threshold)

    st.divider()

    st.subheader('Resultado da Predição')

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(
            label='Probabilidade de Risco',
            value=f'{probabilidade:.2%}'
        )

    with col_b:
        if predicao == 1:
            st.error('Aluno classificado como ALTO RISCO')
        else:
            st.success('Aluno classificado como BAIXO RISCO')

    # =========================
    # GRÁFICO DE PROBABILIDADE
    # =========================

    st.subheader('Probabilidade do Modelo')

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(['Risco'], [probabilidade])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probabilidade')
    ax.set_title('Probabilidade Prevista')

    st.pyplot(fig)

    # =========================
    # IMPORTÂNCIA DAS FEATURES
    # =========================

    st.subheader('Importância das Variáveis')

    importancias = pd.DataFrame({
        'Variavel': features,
        'Importancia': model.feature_importances_
    })

    importancias = importancias.sort_values(
        by='Importancia',
        ascending=True
    )

    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.barh(
        importancias['Variavel'],
        importancias['Importancia']
    )

    ax2.set_title('Importância das Features')

    st.pyplot(fig2)

    # =========================
    # DADOS UTILIZADOS
    # =========================

    st.subheader('Dados Informados')

    st.dataframe(novo_aluno)
