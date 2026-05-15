import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# =============================================================================
# TEMA GLOBAL — alinhado ao layout da apresentação Gamma (azul escuro + moderno)
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
[data-testid="stSidebar"] * { color: #CBD5DF !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 { color: #F4A261 !important; }

/* ── Títulos ── */
h1 { color: #F4A261 !important; letter-spacing: -0.5px; }
h2, h3 { color: #7EC8E3 !important; }

/* ── Métricas (st.metric) ── */
[data-testid="stMetric"] {
    background: var(--pm-card-bg);
    border: 1px solid var(--pm-border);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"]  { color: #8AAFC7 !important; font-size: 0.78rem; }
[data-testid="stMetricValue"]  { color: #F4A261 !important; font-weight: 700; }
[data-testid="stMetricDelta"]  { color: #7EC8E3 !important; }

/* ── Dataframe / tabelas ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--pm-border);
    border-radius: 8px;
    overflow: hidden;
}

/* ── Selectbox / radio / slider ── */
[data-testid="stSelectbox"] > div,
[data-testid="stRadio"] > div {
    background: var(--pm-card-bg);
    border-radius: 8px;
    border: 1px solid var(--pm-border);
}

/* ── Botões ── */
.stButton > button {
    background: linear-gradient(135deg, #1B4F72, #2E86C1);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Dividers ── */
hr { border-color: var(--pm-border) !important; }

/* ── Info / Warning / Error boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 4px;
}

/* ── Caption / small text ── */
[data-testid="stCaptionContainer"] { color: #8AAFC7 !important; }

/* ── Apresentação fullscreen container ── */
.pm-presentation-wrapper {
    position: relative;
    width: 100%;
    padding-top: 56.25%;   /* 16:9 */
    border-radius: 14px;
    overflow: hidden;
    border: 2px solid var(--pm-border);
    box-shadow: 0 8px 40px rgba(0,0,0,0.55);
}
.pm-presentation-wrapper iframe {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    border: none;
}

/* ── Fullscreen button ── */
.pm-fs-btn {
    display: inline-block;
    margin-top: 14px;
    padding: 10px 24px;
    background: linear-gradient(135deg, #1B4F72, #2E86C1);
    color: #fff !important;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.92rem;
    transition: opacity .2s;
    cursor: pointer;
}
.pm-fs-btn:hover { opacity: 0.82; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONSTANTES
# =============================================================================
CORES = {
    "primaria":   "#1B4F72",
    "secundaria": "#2E86C1",
    "destaque":   "#E74C3C",
    "verde":      "#1E8449",
    "amarelo":    "#F39C12",
    "cinza":      "#7F8C8D",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})





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

model_path = os.path.join(BASE_DIR, 'modelo.pkl')

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


# ============================================
# NOME + CONFIGURAÇÕES
# ============================================

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



# ============================================
# CAMPOS NUMÉRICOS
# ============================================

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
                st.error("ALTO RISCO")
            else:
                st.success("BAIXO RISCO")

        # ====================================
        # GRÁFICO RISCO
        # ====================================

        st.subheader("Nível de Risco")

        fig, ax = plt.subplots(figsize=(6, 3))

        ax.barh(
            ['Risco'],
            [probabilidade]
        )

        ax.set_xlim(0, 1)

        ax.set_xlabel('Probabilidade')

        st.pyplot(fig)

        # ====================================
        # IMPORTÂNCIA FEATURES
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

        dados_exibicao = novo_registro.copy()

        dados_exibicao.insert(
            0,
            'Nome_Aluno',
            nome_aluno
        )

        st.dataframe(dados_exibicao)

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
        # HISTOGRAMA
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
