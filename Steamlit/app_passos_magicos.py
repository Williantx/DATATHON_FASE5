import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Modelo Preditivo",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# TEMA GLOBAL
# =============================================================================

st.markdown("""
<style>

/* ── Paleta base ── */
:root {
    --pm-blue-deep:   #0D1B2A;
    --pm-blue-mid:    #1B4F72;
    --pm-blue-light:  #2E86C1;
    --pm-accent:      #F4A261;
    --pm-white:       #F8F9FA;
    --pm-card-bg:     #132233;
    --pm-border:      rgba(46,134,193,0.35);
}

/* ── Fundo principal ── */
.stApp {
    background: linear-gradient(160deg, #0D1B2A 0%, #102030 60%, #0D1B2A 100%);
    color: #E8EDF2;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0f2236 100%);
    border-right: 1px solid var(--pm-border);
}

[data-testid="stSidebar"] * {
    color: #CBD5DF !important;
}

/* ── Títulos ── */
h1 {
    color: #F4A261 !important;
}

h2, h3 {
    color: #7EC8E3 !important;
}

/* ── Cards ── */
[data-testid="stMetric"] {
    background: #132233;
    border: 1px solid rgba(46,134,193,0.35);
    border-radius: 12px;
    padding: 16px;
}

/* ── Botões ── */
.stButton > button {
    background: linear-gradient(135deg, #1B4F72, #2E86C1);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    height: 45px;
    width: 100%;
}

.stButton > button:hover {
    opacity: 0.90;
}

/* ── Inputs ── */
.stTextInput input,
.stNumberInput input {
    background-color: #132233;
    color: white;
    border-radius: 8px;
}

/* ── Slider ── */
.stSlider {
    padding-top: 10px;
}

/* ── Tabela ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURAÇÕES DO MATPLOTLIB
# =============================================================================

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =============================================================================
# CAMINHO DO MODELO
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'modelo.pkl')

# =============================================================================
# CARREGAR MODELO
# =============================================================================

try:
    model = joblib.load(model_path)

except Exception as e:
    st.error(f'Erro ao carregar modelo: {e}')
    st.stop()

# =============================================================================
# FEATURES DO MODELO
# =============================================================================

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

# =============================================================================
# HEADER
# =============================================================================

st.title("📊 Modelo Preditivo - Passos Mágicos")

st.markdown("""
Sistema de análise preditiva utilizando Machine Learning
para classificação de risco de alunos.
""")

st.divider()

# =============================================================================
# FORMULÁRIO
# =============================================================================

st.subheader("Inserir Dados do Aluno")

# =============================================================================
# NOME + THRESHOLD
# =============================================================================

col_nome, col_threshold = st.columns([2, 1])

with col_nome:

    nome_aluno = st.text_input(
        "Nome do Aluno",
        placeholder="Digite o nome do aluno"
    )

with col_threshold:

    threshold = st.slider(
        "Threshold de Risco",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01
    )

# =============================================================================
# CAMPOS NUMÉRICOS
# =============================================================================

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

# =============================================================================
# DATAFRAME MODELO
# =============================================================================

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

# =============================================================================
# BOTÃO
# =============================================================================

if st.button("🔍 Gerar Análise"):

    try:

        # =========================================================================
        # PREDIÇÃO
        # =========================================================================

        probabilidade = model.predict_proba(
            novo_registro
        )[0][1]

        predicao = int(probabilidade >= threshold)

        st.divider()

        # =========================================================================
        # RESULTADO
        # =========================================================================

        st.subheader("Resultado da Análise")

        if nome_aluno:
            st.info(f"👤 Aluno analisado: {nome_aluno}")

        colA, colB = st.columns(2)

        with colA:

            st.metric(
                "Probabilidade de Risco",
                f"{probabilidade:.2%}"
            )

        with colB:

            if predicao == 1:

                st.markdown(
                    '''
                    <div style="
                        background-color:#641E16;
                        padding:20px;
                        border-radius:10px;
                        text-align:center;
                        font-size:28px;
                        font-weight:bold;
                        color:white;">
                        🚨 ALTO RISCO
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '''
                    <div style="
                        background-color:#145A32;
                        padding:20px;
                        border-radius:10px;
                        text-align:center;
                        font-size:28px;
                        font-weight:bold;
                        color:white;">
                        ✅ BAIXO RISCO
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

        # =========================================================================
        # GRÁFICO DE RISCO
        # =========================================================================

        st.subheader("Nível de Risco")

        fig, ax = plt.subplots(figsize=(8, 2))

        ax.barh(
            ['Risco'],
            [probabilidade]
        )

        ax.set_xlim(0, 1)

        ax.set_xlabel('Probabilidade')

        st.pyplot(fig)

        # =========================================================================
        # IMPORTÂNCIA FEATURES
        # =========================================================================

        if hasattr(model, 'feature_importances_'):

            st.subheader("Importância das Variáveis")

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

        # =========================================================================
        # DADOS UTILIZADOS
        # =========================================================================

        st.subheader("Dados Informados")

        dados_exibicao = novo_registro.copy()

        dados_exibicao.insert(
            0,
            'Nome_Aluno',
            nome_aluno
        )

        st.dataframe(dados_exibicao)

    except Exception as e:

        st.error(f'Erro na previsão: {e}')

# =============================================================================
# UPLOAD CSV
# =============================================================================

st.divider()

st.subheader("📂 Análise em Lote")

arquivo = st.file_uploader(
    "Faça upload de um CSV",
    type=['csv']
)

if arquivo is not None:

    try:

        df = pd.read_csv(arquivo)

        st.write("Prévia dos dados")

        st.dataframe(df.head())

        # =========================================================================
        # GARANTIR FEATURES CORRETAS
        # =========================================================================

        df_modelo = df[features]

        probabilidades = model.predict_proba(
            df_modelo
        )[:, 1]

        df['Probabilidade_Risco'] = probabilidades

        df['Classificacao'] = np.where(
            df['Probabilidade_Risco'] >= threshold,
            'ALTO RISCO',
            'BAIXO RISCO'
        )

        st.subheader("Resultado")

        st.dataframe(df)

        # =========================================================================
        # HISTOGRAMA
        # =========================================================================

        st.subheader("Distribuição de Probabilidades")

        fig3, ax3 = plt.subplots(figsize=(8, 5))

        ax3.hist(
            df['Probabilidade_Risco'],
            bins=10
        )

        ax3.set_xlabel('Probabilidade')

        ax3.set_ylabel('Quantidade')

        st.pyplot(fig3)

        # =========================================================================
        # DOWNLOAD CSV
        # =========================================================================

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Resultado CSV",
            data=csv,
            file_name='resultado_analise.csv',
            mime='text/csv'
        )

    except Exception as e:

        st.error(f'Erro no processamento do CSV: {e}')
