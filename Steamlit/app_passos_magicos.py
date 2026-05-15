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
    page_title="Analytics - Passos Mágicos",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# TEMA GLOBAL CUSTOMIZADO
# =============================================================================
st.markdown("""
<style>
    :root {
        --pm-blue-deep:   #0D1B2A;
        --pm-blue-mid:    #1B4F72;
        --pm-blue-light:  #2E86C1;
        --pm-accent:      #F4A261;
        --pm-white:       #F8F9FA;
    }
    .stApp {
        background: linear-gradient(160deg, #0D1B2A 0%, #102030 60%, #0D1B2A 100%);
        color: #E8EDF2;
    }
    [data-testid="stSidebar"] {
        background: #0D1B2A;
        border-right: 1px solid rgba(46,134,193,0.3);
    }
    h1 { color: #F4A261 !important; }
    h2, h3 { color: #7EC8E3 !important; }
    .stMetric {
        background: #132233;
        border: 1px solid rgba(46,134,193,0.35);
        border-radius: 12px;
        padding: 15px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1B4F72, #2E86C1);
        color: white !important;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CARREGAMENTO DO MODELO E CONFIGURAÇÕES FIXAS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'modelo.pkl')

# DEFINIÇÃO DO THRESHOLD FIXO CONFORME SOLICITADO
THRESHOLD_FIXO = 0.70

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f'Erro ao carregar modelo: {e}')
    st.stop()

features = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'Fase_Num', 'IPV', 'Anos_No_Programa']

# =============================================================================
# HEADER PRINCIPAL
# =============================================================================
st.title("📊 Plataforma Analytics - Passos Mágicos")
st.divider()

# Navegação por Abas
tab1, tab2, tab3 = st.tabs(["🔍 Predição de Risco", "📽️ Apresentação", "⚙️ Análise Técnica"])

# =============================================================================
# ABA 1: PREDIÇÃO (COM THRESHOLD TRAVADO EM 0.70)
# =============================================================================
with tab1:
    st.subheader("Inserir Dados do Aluno")
    
    # Campo de nome ocupa a largura total agora que o slider foi removido
    nome_aluno = st.text_input("Nome do Aluno", placeholder="Digite o nome completo")
    st.caption(f"ℹ️ O sistema utiliza um Threshold de Risco padrão de {THRESHOLD_FIXO}")

    c1, c2, c3, c4 = st.columns(4)
    IDA = c1.number_input("IDA", value=7.0)
    IEG = c2.number_input("IEG", value=7.0)
    IAA = c3.number_input("IAA", value=7.0)
    IPS = c4.number_input("IPS", value=7.0)

    c5, c6, c7, c8 = st.columns(4)
    IPP = c5.number_input("IPP", value=7.0)
    Fase_Num = c6.number_input("Fase_Num", value=3)
    IPV = c7.number_input("IPV", value=7.0)
    Anos_No_Programa = c8.number_input("Anos_No_Programa", value=2)

    novo_registro = pd.DataFrame([[IDA, IEG, IAA, IPS, IPP, Fase_Num, IPV, Anos_No_Programa]], columns=features)

    if st.button("🔍 Gerar Análise"):
        probabilidade = model.predict_proba(novo_registro)[0][1]
        
        # Uso do valor fixo para a predição
        predicao = int(probabilidade >= THRESHOLD_FIXO)

        st.divider()
        colA, colB = st.columns(2)
        with colA:
            st.metric("Probabilidade de Risco", f"{probabilidade:.2%}")
        with colB:
            cor = "#641E16" if predicao == 1 else "#145A32"
            status = "🚨 ALTO RISCO" if predicao == 1 else "✅ BAIXO RISCO"
            st.markdown(f'<div style="background:{cor};padding:20px;border-radius:10px;text-align:center;font-size:24px;font-weight:bold;color:white;">{status}</div>', unsafe_allow_html=True)

# =============================================================================
# ABA 2: INTRODUÇÃO E APRESENTAÇÃO
# =============================================================================
with tab2:
    st.header("Introdução do Projeto")
    st.markdown("""
    ### Transformando Dados em Oportunidades
    Este projeto utiliza modelos preditivos para identificar alunos em situação de vulnerabilidade acadêmica ou psicossocial. 
    O objetivo é permitir que a equipe pedagógica atue de forma proativa.
    
    *   **Metodologia:** Classificação baseada em indicadores de desempenho e engajamento.
    *   **Foco:** Garantir que nenhum aluno fique para trás na jornada de aprendizado.
    """)
    
    st.divider()
    
    st.subheader("📽️ Apresentação Executiva")
    ppt_file = os.path.join(BASE_DIR, "apresentacao.pptx")
    if os.path.exists(ppt_file):
        with open(ppt_file, "rb") as f:
            st.download_button(
                label="📥 Baixar Apresentação (PowerPoint)",
                data=f,
                file_name="Apresentacao_Passos_Magicos.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
    else:
        st.warning("Arquivo 'apresentacao.pptx' não encontrado.")

# =============================================================================
# ABA 3: ANÁLISE TÉCNICA
# =============================================================================
with tab3:
    st.header("⚙️ Detalhes do Modelo")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.write("**Parâmetros de Decisão:**")
        st.write(f"- Threshold de Corte: **{THRESHOLD_FIXO}**")
        st.write("- Algoritmo: Random Forest / Gradient Boosting")
    
    with col_t2:
        st.write("**Indicadores Analisados:**")
        st.caption("IDA, IEG, IAA, IPS, IPP, Fase, IPV e Tempo de Programa.")

    st.divider()
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Importância das Variáveis")
        importancia = pd.DataFrame({'Variavel': features, 'Valor': model.feature_importances_}).sort_values('Valor')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(importancia['Variavel'], importancia['Valor'], color='#F4A261')
        st.pyplot(fig)
