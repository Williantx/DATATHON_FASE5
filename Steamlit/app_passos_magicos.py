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
        color: #FFFFFF !important;
    }
    
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp div {
        color: #FFFFFF !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    button[data-baseweb="tab"] p {
        color: #FFFFFF !important;
    }
    
    .stMetric {
        background: #132233;
        border: 1px solid rgba(46,134,193,0.35);
        border-radius: 12px;
        padding: 15px;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #FFFFFF !important;
    }
    
    .stTextInput input, .stNumberInput input {
        color: #FFFFFF !important;
        background-color: #132233 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1B4F72, #2E86C1);
        color: #FFFFFF !important;
        font-weight: 600;
        width: 100%;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CARREGAMENTO DO MODELO E CONFIGURAÇÃO
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'modelo.pkl')

# ATUALIZADO: Threshold para 0.70 conforme solicitado
THRESHOLD_FIXO = 0.70

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f'Erro ao carregar modelo: {e}')
    st.stop()

features = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'Fase_Num', 'IPV', 'Anos_No_Programa']

# =============================================================================
# HEADER
# =============================================================================
st.title("📊 Plataforma Analytics - Passos Mágicos")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Predição de Risco", 
    "📽️ Apresentação", 
    "⚙️ Análise Técnica",
    "🎥 Vídeo Demonstrativo"
])

# =============================================================================
# ABA 1: PREDIÇÃO
# =============================================================================
with tab1:
    st.subheader("Inserir Dados do Aluno")
    nome_aluno = st.text_input("Nome do Aluno", placeholder="Digite o nome completo")
    st.caption(f"ℹ️ Threshold de Risco fixado em {THRESHOLD_FIXO}")

    c1, c2, c3, c4 = st.columns(4)
    IDA = c1.number_input("IDA", value=7.0, min_value=0.0, max_value=10.0)
    IEG = c2.number_input("IEG", value=7.0, min_value=0.0, max_value=10.0)
    IAA = c3.number_input("IAA", value=7.0, min_value=0.0, max_value=10.0)
    IPS = c4.number_input("IPS", value=7.0, min_value=0.0, max_value=10.0)

    c5, c6, c7, c8 = st.columns(4)
    IPP = c5.number_input("IPP", value=7.0, min_value=0.0, max_value=10.0)
    Fase_Num = c6.number_input("Fase_Num", value=3)
    IPV = c7.number_input("IPV", value=7.0, min_value=0.0, max_value=10.0)
    Anos_No_Programa = c8.number_input("Anos_No_Programa", value=2)

    if st.button("🔍 Gerar Análise"):
        novo_registro = pd.DataFrame([[IDA, IEG, IAA, IPS, IPP, Fase_Num, IPV, Anos_No_Programa]], columns=features)
        
        # Predição de probabilidade
        probabilidade = model.predict_proba(novo_registro)[0][1]
        
        st.divider()
        
        if nome_aluno:
            st.markdown(f"### 👤 Aluno analisado: **{nome_aluno}**")
        else:
            st.markdown("### 👤 Aluno analisado: *Nome não informado*")

        colA, colB = st.columns(2)
        with colA:
            st.metric("Probabilidade de Risco", f"{probabilidade:.2%}")
        
        with colB:
            # Lógica corrigida: Se >= 0.70 é ALTO RISCO (Vermelho)
            if probabilidade >= THRESHOLD_FIXO:
                cor = "#641E16" # Vermelho Escuro
                status = "🚨 ALTO RISCO"
                st.markdown(f'<div style="background:{cor};padding:20px;border-radius:10px;text-align:center;font-size:24px;font-weight:bold;color:white;">{status}</div>', unsafe_allow_html=True)
                st.warning("Recomendação: Intervenção pedagógica imediata e acompanhamento psicossocial.")
            else:
                cor = "#145A32" # Verde Escuro
                status = "✅ SEM RISCO IMEDIATO"
                st.markdown(f'<div style="background:{cor};padding:20px;border-radius:10px;text-align:center;font-size:24px;font-weight:bold;color:white;">{status}</div>', unsafe_allow_html=True)
                st.info("O aluno apresenta indicadores estáveis. Manter monitoramento de rotina.")

# =============================================================================
# ABA 2: APRESENTAÇÃO
# =============================================================================
with tab2:
    st.header("Dados que Transformam Vidas")
    st.markdown("""
    ### Inteligência Educacional Preventiva
    Este ecossistema visa identificar alunos em risco de defasagem antes que o problema se consolide.
    """)
    st.divider()
    slides_url = "https://docs.google.com/presentation/d/19KuBSyKADQBgzwnsW4526j6pSOXu2c8D/embed"
    st.components.v1.iframe(slides_url, height=550)

# =============================================================================
# ABA 3: ANÁLISE TÉCNICA
# =============================================================================
with tab3:
    st.header("⚙️ Engenharia e Machine Learning")
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Importância das Variáveis")
        importancia = pd.DataFrame({'Variavel': features, 'Valor': model.feature_importances_}).sort_values('Valor')
        
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0D1B2A')
        ax.set_facecolor('#132233')
        ax.barh(importancia['Variavel'], importancia['Valor'], color='#2E86C1')
        
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        for spine in ax.spines.values():
            spine.set_color('white')
            
        st.pyplot(fig)

# =============================================================================
# ABA 4: VÍDEO
# =============================================================================
with tab4:
    st.header("🎥 Vídeo de Apresentação")
    youtube_url = "https://youtu.be/Fqq_1ExsETw?si=XenMar6fN2v6cjbW"
    st.video(youtube_url)
