```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================

st.set_page_config(
    page_title="Modelo Preditivo",
    page_icon="📊",
    layout="wide"
)

# ============================================
# CAMINHO DO MODELO
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'models', 'modelo.pkl')

# ============================================
# CARREGAR MODELO
# ============================================

try:
    model = joblib.load(model_path)

except Exception as e:
    st.error(f'Erro ao carregar modelo: {e}')
    st.stop()

# ============================================
# TÍTULO
# ============================================

st.title("📊 Modelo Preditivo - Passos Mágicos")

st.markdown("""
Sistema de análise preditiva utilizando Machine Learning.
""")

# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("Configurações")

threshold = st.sidebar.slider(
    "Threshold de Risco",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.01
)

# ============================================
# FEATURES DO MODELO
# ============================================

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

# ============================================
# FORMULÁRIO
# ============================================

st.subheader("Inserir Dados")

col1, col2, col3, col4 = st.columns(4)

with col1:
    IDA = st.number_input("IDA", value=7.0)

with col2:
    IEG = st.number_input("IEG", value=7.0)

with col3:
    IAA = st.number_input("IAA", value=7.0)

with col4:
    IPS = st.number_input("IPS", value=7.0)

col5, col6, col7, col8 = st.columns(4)

with col5:
    IPP = st.number_input("IPP", value=7.0)

with col6:
    Fase_Num = st.number_input("Fase_Num", value=3)

with col7:
    IPV = st.number_input("IPV", value=7.0)

with col8:
    Anos_No_Programa = st.number_input(
        "Anos_No_Programa",
        value=2
    )

# ============================================
# DATAFRAME
# ============================================

novo_registro = pd.DataFrame({
    'IDA': [IDA],
    'IEG': [IEG],
    'IAA': [IAA],
    'IPS': [IPS],
    'IPP': [IPP],
    'Fase_Num': [Fase_Num],
    'IPV': [IPV],
    'Anos_No_Programa': [Anos_No_Programa]
})

# ============================================
# BOTÃO PREDIÇÃO
# ============================================

if st.button("Gerar Análise"):

    try:

        # ====================================
        # PREVISÃO
        # ====================================

        probabilidade = model.predict_proba(
            novo_registro
        )[0][1]

        predicao = int(probabilidade >= threshold)

        # ====================================
        # RESULTADOS
        # ====================================

        st.divider()

        st.subheader("Resultado da Análise")

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                "Probabilidade de Risco",
                f"{probabilidade:.2%}"
            )

        with colB:

            if predicao == 1:
                st.error("ALTO RISCO")
            else:
                st.success("BAIXO RISCO")

        # ====================================
        # GAUGE
        # ====================================

        st.subheader("Nível de Risco")

        fig, ax = plt.subplots(figsize=(6, 3))

        ax.barh(
            ['Risco'],
            [probabilidade]
        )

        ax.set_xlim(0, 1)

        st.pyplot(fig)

        # ====================================
        # IMPORTÂNCIA DAS FEATURES
        # ====================================

        if hasattr(model, 'feature_importances_'):

            st.subheader(
                "Importância das Variáveis"
            )

            importancia = pd.DataFrame({
                'Variavel': features,
                'Importancia': model.feature_importances_
            })

            importancia = importancia.sort_values(
                by='Importancia',
                ascending=True
            )

            fig2, ax2 = plt.subplots(figsize=(8, 5))

            ax2.barh(
                importancia['Variavel'],
                importancia['Importancia']
            )

            ax2.set_xlabel('Importância')

            st.pyplot(fig2)

        # ====================================
        # DADOS UTILIZADOS
        # ====================================

        st.subheader("Dados Informados")

        st.dataframe(novo_registro)

    except Exception as e:
        st.error(f'Erro na previsão: {e}')

# ============================================
# UPLOAD CSV
# ============================================

st.divider()

st.subheader("Análise em Lote")

arquivo = st.file_uploader(
    "Faça upload de um CSV",
    type=['csv']
)

if arquivo is not None:

    try:

        df = pd.read_csv(arquivo)

        st.write("Prévia dos dados")

        st.dataframe(df.head())

        probabilidades = model.predict_proba(df)[:, 1]

        df['Probabilidade_Risco'] = probabilidades

        df['Classificacao'] = np.where(
            df['Probabilidade_Risco'] >= threshold,
            'ALTO RISCO',
            'BAIXO RISCO'
        )

        st.subheader("Resultado")

        st.dataframe(df)

        # ====================================
        # GRÁFICO DISTRIBUIÇÃO
        # ====================================

        st.subheader("Distribuição de Probabilidades")

        fig3, ax3 = plt.subplots(figsize=(8, 5))

        ax3.hist(
            df['Probabilidade_Risco'],
            bins=10
        )

        ax3.set_xlabel('Probabilidade')

        ax3.set_ylabel('Quantidade')

        st.pyplot(fig3)

        # ====================================
        # DOWNLOAD CSV
        # ====================================

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Resultado CSV",
            data=csv,
            file_name='resultado_analise.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f'Erro no processamento do CSV: {e}')
```
