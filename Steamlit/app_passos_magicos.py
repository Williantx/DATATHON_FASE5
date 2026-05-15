import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, roc_curve,
    precision_score, recall_score
)


# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Analytics - Passos Mágicos",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# TEMA GLOBAL CUSTOMIZADO (TODAS AS LETRAS EM BRANCO)
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
# CONSTANTES
# =============================================================================
FEATURES       = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'Fase_Num', 'IPV', 'Anos_No_Programa']
THRESHOLD_FIXO = 0.70
FILE_URL = (
    'https://docs.google.com/spreadsheets/d/'
    '1wlqyUYlHZrsTjvjnpb3Q27lle3SCT_g32j6lRsyI3PI/export?format=xlsx'
)
BG_DARK  = '#0D1B2A'
BG_CARD  = '#132233'
CLR_BLUE = '#2E86C1'
CLR_AMB  = '#F4A261'
CLR_W    = '#FFFFFF'

# =============================================================================
# ETL — igual ao notebook
# =============================================================================
@st.cache_data(show_spinner="⏳ Carregando e processando dados da planilha…")
def carregar_e_processar():
    df_2022 = pd.read_excel(FILE_URL, sheet_name='2022')
    df_2023 = pd.read_excel(FILE_URL, sheet_name='2023')
    df_2024 = pd.read_excel(FILE_URL, sheet_name='2024')

    rename_common = {
        'Nome Anonimizado': 'Nome', 'Data de Nasc': 'Data_Nasc',
        'Gênero': 'Genero', 'Ano ingresso': 'Ano_Ingresso',
        'Instituição de ensino': 'Instituicao_Ensino',
        'Nº Av': 'Num_Avaliacoes', 'Rec Av1': 'Rec_Av1',
        'Rec Av2': 'Rec_Av2', 'Rec Psicologia': 'Rec_Psicologia',
        'Indicado': 'Indicado_Bolsa', 'Atingiu PV': 'Atingiu_PV',
        'Destaque IEG': 'Destaque_IEG', 'Destaque IDA': 'Destaque_IDA',
        'Destaque IPV': 'Destaque_IPV', 'Mat': 'Nota_Mat',
        'Por': 'Nota_Por', 'Ing': 'Nota_Ing',
    }
    COLUNAS_FINAIS = [
        'RA', 'Ano', 'Fase', 'Turma', 'Nome', 'Data_Nasc', 'Idade', 'Genero',
        'Ano_Ingresso', 'Instituicao_Ensino', 'INDE', 'Pedra',
        'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV',
        'Nota_Mat', 'Nota_Por', 'Nota_Ing', 'Cg', 'Cf', 'Ct',
        'Num_Avaliacoes', 'Indicado_Bolsa', 'Atingiu_PV',
        'Fase_Ideal', 'Defasagem', 'Destaque_IEG', 'Destaque_IDA',
        'Destaque_IPV', 'Rec_Psicologia',
    ]

    def padronizar_aba(df, rename_esp, ano):
        ren = {**rename_common, **rename_esp}
        ren_val = {k: v for k, v in ren.items() if k in df.columns}
        df_r = df.rename(columns=ren_val)
        df_r['Ano'] = ano
        cols = [c for c in COLUNAS_FINAIS if c in df_r.columns]
        return df_r[cols]

    fase_map = {0:'ALFA',1:'FASE 1',2:'FASE 2',3:'FASE 3',
                4:'FASE 4',5:'FASE 5',6:'FASE 6',7:'FASE 7',8:'FASE 8'}

    df22 = padronizar_aba(df_2022,
        {'INDE 22':'INDE','Pedra 22':'Pedra','Fase ideal':'Fase_Ideal','Defas':'Defasagem'}, 2022)
    df23 = padronizar_aba(df_2023,
        {'INDE 2023':'INDE','Pedra 2023':'Pedra','Fase Ideal':'Fase_Ideal'}, 2023)
    df24 = padronizar_aba(df_2024,
        {'INDE 2024':'INDE','Pedra 2024':'Pedra','Fase Ideal':'Fase_Ideal',
         'Ativo/ Inativo':'Status_Ativo'}, 2024)

    df22['Fase'] = df22['Fase'].map(fase_map).fillna(df22['Fase'].astype(str))

    def norm_fase(f):
        if f in ['ALFA','FASE 1','FASE 2','FASE 3','FASE 4',
                 'FASE 5','FASE 6','FASE 7','FASE 8']:
            return f
        m = re.match(r'^(\d+)', str(f))
        if m:
            n = int(m.group(1))
            if n == 0: return 'ALFA'
            if 1 <= n <= 8: return f'FASE {n}'
            if n == 9: return 'FASE 8'
        return f

    for dt in [df22, df23, df24]:
        dt['Fase'] = dt['Fase'].astype(str).str.strip().str.upper().apply(norm_fase)

    df_all = pd.concat([df22, df23, df24], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['RA','Ano'], keep='first')

    num_cols = ['IAN','IDA','IEG','IAA','IPS','IPP','IPV',
                'Nota_Mat','Nota_Por','Nota_Ing','Cg','Cf','Ct',
                'Indicado_Bolsa','Atingiu_PV','Defasagem','Num_Avaliacoes','INDE']
    for c in num_cols:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

    pedra_fix = {
        'Agata':'Ágata','agata':'Ágata','AGATA':'Ágata','ÁGATA':'Ágata',
        'QUARTZO':'Quartzo','AMETISTA':'Ametista',
        'TOPÁZIO':'Topázio','TOPAZIO':'Topázio'
    }
    if 'Pedra' in df_all.columns:
        df_all['Pedra'] = df_all['Pedra'].replace(pedra_fix)
        df_all['Pedra'] = pd.Categorical(
            df_all['Pedra'],
            categories=['Quartzo','Ágata','Ametista','Topázio'], ordered=True)

    if 'Genero' in df_all.columns:
        df_all['Genero'] = df_all['Genero'].astype(str).str.strip().str.capitalize()
        df_all['Genero'] = df_all['Genero'].replace({'Menino':'Masculino','Menina':'Feminino'})

    df_all['Anos_No_Programa'] = (df_all['Ano'] - df_all['Ano_Ingresso']).clip(lower=0)
    df_all['Risco_Defasagem']  = (df_all['Defasagem'] < 0).astype(int)
    df_all['Pedra_Num']        = df_all['Pedra'].map(
        {'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4})

    def fase_num(f):
        if pd.isna(f): return np.nan
        s = str(f).upper()
        if 'ALFA' in s: return 0
        for i in range(9, 0, -1):
            if str(i) in s: return i
        return np.nan

    df_all['Fase_Num'] = df_all['Fase'].apply(fase_num)

    # Target: INDE abaixo da mediana = em risco
    # INDE é o índice geral — quanto menor, pior. Indicadores baixos → risco=1
    thr_inde = df_all["INDE"].median()
    df_all['risco_modelo'] = (df_all['INDE'] < thr_inde).astype(int)

    return df_all


# =============================================================================
# TREINAMENTO — idêntico ao notebook
# =============================================================================
@st.cache_resource(show_spinner="🤖 Treinando modelo Random Forest…")
def treinar_modelo(df_all):
    df_m = df_all.dropna(subset=FEATURES + ['risco_modelo']).copy()
    X = df_m[FEATURES]
    y = df_m['risco_modelo']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    # class_weight='balanced' compensa o desbalanceamento sem depender do imblearn
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba > THRESHOLD_FIXO).astype(int)

    return model, X_test, y_test, proba, pred


# =============================================================================
# HELPER — figura com fundo escuro
# =============================================================================
def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG_DARK)
    ax.set_facecolor(BG_CARD)
    for sp in ax.spines.values():
        sp.set_color('#2E4A62')
    ax.tick_params(colors=CLR_W)
    ax.xaxis.label.set_color(CLR_W)
    ax.yaxis.label.set_color(CLR_W)
    ax.title.set_color(CLR_W)
    return fig, ax


# =============================================================================
# INICIALIZAR DADOS E MODELO
# =============================================================================
try:
    df_all = carregar_e_processar()
    model, X_test, y_test, proba_test, pred_test = treinar_modelo(df_all)
except Exception as e:
    st.error(f"Erro ao carregar dados ou treinar o modelo: {e}")
    st.info("Verifique se a planilha do Google Sheets está com acesso público de leitura.")
    st.stop()

roc_score = roc_auc_score(y_test, proba_test)
medias    = df_all[FEATURES].mean()

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
# ABA 1 — PREDIÇÃO DE RISCO
# =============================================================================
with tab1:
    st.subheader("Inserir Dados do Aluno")
    nome_aluno = st.text_input("Nome do Aluno", placeholder="Digite o nome completo")
    st.caption(f"ℹ️ Threshold de Risco fixado em {THRESHOLD_FIXO:.0%}")

    c1, c2, c3, c4 = st.columns(4)
    IDA              = c1.number_input("IDA",              min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    IEG              = c2.number_input("IEG",              min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    IAA              = c3.number_input("IAA",              min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    IPS              = c4.number_input("IPS",              min_value=0.0, max_value=10.0, value=7.0, step=0.1)

    c5, c6, c7, c8 = st.columns(4)
    IPP              = c5.number_input("IPP",              min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    Fase_Num         = c6.number_input("Fase_Num",         min_value=0,   max_value=8,    value=3,   step=1)
    IPV              = c7.number_input("IPV",              min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    Anos_No_Programa = c8.number_input("Anos_No_Programa", min_value=0,   max_value=15,   value=2,   step=1)

    if st.button("🔍 Gerar Análise"):
        novo = pd.DataFrame(
            [[IDA, IEG, IAA, IPS, IPP, Fase_Num, IPV, Anos_No_Programa]],
            columns=FEATURES
        )
        probabilidade = model.predict_proba(novo)[0][1]
        predicao      = int(probabilidade >= THRESHOLD_FIXO)

        st.divider()

        # Nome do aluno
        if nome_aluno:
            st.markdown(f"### 👤 Aluno analisado: **{nome_aluno}**")
        else:
            st.markdown("### 👤 Aluno analisado: *Nome não informado*")

        # Métricas principais
        colA, colB = st.columns(2)
        with colA:
            st.metric("Probabilidade de Risco", f"{probabilidade:.2%}")
        with colB:
            cor    = "#641E16" if predicao == 1 else "#145A32"
            status = "🚨 ALTO RISCO" if predicao == 1 else "✅ BAIXO RISCO"
            st.markdown(
                f'<div style="background:{cor};padding:20px;border-radius:10px;'
                f'text-align:center;font-size:24px;font-weight:bold;color:white;">'
                f'{status}</div>', unsafe_allow_html=True)

        # Barra de probabilidade
        st.markdown("#### Nível de Risco")
        bar_color = '#d32f2f' if probabilidade >= 0.70 else '#f57c00' if probabilidade >= 0.45 else '#388e3c'
        fig, ax = dark_fig(8, 1.4)
        ax.barh(0, 1,            height=0.5, color='#1e3a52', zorder=0)
        ax.barh(0, probabilidade, height=0.5, color=bar_color, zorder=1)
        ax.axvline(THRESHOLD_FIXO, color=CLR_AMB, lw=2, linestyle='--',
                   label=f'Threshold ({THRESHOLD_FIXO:.0%})')
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xlabel("Probabilidade de Risco")
        ax.legend(loc='upper right', fontsize=9, facecolor=BG_CARD, labelcolor=CLR_W)
        ax.text(min(probabilidade + 0.02, 0.92), 0,
                f"{probabilidade:.1%}", va='center',
                color=CLR_W, fontsize=12, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        # Comparativo com a base
        st.markdown("#### Comparativo com a Média da Base")
        aluno_vals_list = [IDA, IEG, IAA, IPS, IPP, Fase_Num, IPV, Anos_No_Programa]
        comp = pd.DataFrame({
            'Variável':       FEATURES,
            'Este aluno':     aluno_vals_list,
            'Média da base':  medias.values.round(2),
        })
        comp['Diferença'] = (comp['Este aluno'] - comp['Média da base']).round(2)
        comp['']          = comp['Diferença'].apply(lambda x: '▲' if x > 0 else ('▼' if x < 0 else '─'))
        st.dataframe(comp.set_index('Variável'), use_container_width=True)

        # Radar — aluno vs média (apenas indicadores 0-10)
        radar_feats = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']
        vals_aluno  = [IDA, IEG, IAA, IPS, IPP, IPV]
        vals_media  = [float(medias[f]) for f in radar_feats]
        N = len(radar_feats)
        angles = [n / N * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        vals_aluno += vals_aluno[:1]
        vals_media += vals_media[:1]

        fig2, ax2 = plt.subplots(figsize=(5, 5),
                                 subplot_kw=dict(polar=True), facecolor=BG_DARK)
        ax2.set_facecolor(BG_CARD)
        ax2.plot(angles, vals_aluno,  color=CLR_BLUE, lw=2, label='Aluno')
        ax2.fill(angles, vals_aluno,  color=CLR_BLUE, alpha=0.25)
        ax2.plot(angles, vals_media,  color=CLR_AMB,  lw=2, linestyle='--', label='Média')
        ax2.fill(angles, vals_media,  color=CLR_AMB,  alpha=0.10)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(radar_feats, color=CLR_W, size=10)
        ax2.tick_params(colors=CLR_W)
        ax2.set_ylim(0, 10)
        ax2.grid(color='#2E4A62')
        ax2.set_title("Perfil do Aluno vs Média da Base", color=CLR_W, pad=15)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                   facecolor=BG_CARD, labelcolor=CLR_W)
        st.pyplot(fig2); plt.close()

# =============================================================================
# ABA 2 — APRESENTAÇÃO (Google Slides)
# =============================================================================
with tab2:
    st.header("Dados que Transformam Vidas")
    st.markdown("""
    ### Inteligência Educacional Preventiva
    Este ecossistema visa identificar alunos em risco de defasagem antes que o problema se consolide.

    *   **Base de Dados:** 3.030 registros acompanhados entre 2022 e 2024.
    *   **Escopo:** 34 variáveis multidimensionais (acadêmicas, psicossociais e comportamentais).
    *   **Evolução Real:** Redução da taxa de defasagem de 69.9% (2022) para 46.2% (2024).
    *   **Jornada das Pedras:** Alunos no nível Topázio cresceram de 15.1% para 30.9%.
    """)
    st.divider()
    slides_url = "https://docs.google.com/presentation/d/19KuBSyKADQBgzwnsW4526j6pSOXu2c8D/embed"
    st.components.v1.iframe(slides_url, height=550)

# =============================================================================
# ABA 3 — ANÁLISE TÉCNICA (com gráficos reais do modelo treinado)
# =============================================================================
with tab3:
    st.header("⚙️ Engenharia e Machine Learning")

    # KPIs reais do modelo
    report     = classification_report(y_test, pred_test, output_dict=True)
    precisao   = report['1']['precision']
    recall_val = report['1']['recall']
    f1_val     = report['1']['f1-score']

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC-AUC",             f"{roc_score:.3f}")
    k2.metric("Precisão (classe 1)",  f"{precisao:.2%}")
    k3.metric("Recall (classe 1)",    f"{recall_val:.2%}")
    k4.metric("F1-Score (classe 1)",  f"{f1_val:.2%}")

    st.divider()

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        ### Performance e Validação
        - **Algoritmo:** Random Forest Classifier (300 estimadores, profundidade 8).
        - **Balanceamento:** `class_weight='balanced'` para equalizar classes minoritárias no treino.
        - **Threshold:** 0.70 — calibrado para priorizar precisão e mitigar falsos positivos.
        """)
    with col_t2:
        st.markdown("""
        ### Correlações Relevantes
        - **Engajamento (IEG):** Correlação de **0.74** com o INDE. Motor direto do desempenho.
        - **Acadêmico (IDA):** Correlação de **0.78** com o INDE.
        - **IPP e IDA** são os maiores drivers do Ponto de Virada (IPV).
        """)

    st.divider()

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, pred_test)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG_DARK)
        ax.set_facecolor(BG_CARD)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Sem Risco','Em Risco'],
                    yticklabels=['Sem Risco','Em Risco'],
                    ax=ax, linewidths=0.5, linecolor='#1e3a52')
        ax.set_xlabel("Previsto", color=CLR_W)
        ax.set_ylabel("Real",     color=CLR_W)
        ax.set_title("Modelo Final - Matriz de Confusão", color=CLR_W)
        plt.setp(ax.get_xticklabels(), color=CLR_W)
        plt.setp(ax.get_yticklabels(), color=CLR_W)
        ax.tick_params(colors=CLR_W)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_roc:
        st.subheader("Curva ROC")
        fpr, tpr, _ = roc_curve(y_test, proba_test)
        fig, ax = dark_fig(5, 4)
        ax.plot(fpr, tpr, color=CLR_BLUE, lw=2, label=f'AUC = {roc_score:.3f}')
        ax.plot([0,1],[0,1], '--', color='#4a6fa5', lw=1)
        ax.set_xlabel("Taxa de Falsos Positivos")
        ax.set_ylabel("Taxa de Verdadeiros Positivos")
        ax.set_title("Curva ROC")
        ax.legend(facecolor=BG_CARD, labelcolor=CLR_W)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()

    # Importância das variáveis
    st.subheader("Importância das Variáveis")
    importancia = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = dark_fig(10, 4)
    bars = ax.barh(importancia.index, importancia.values, color=CLR_BLUE, height=0.6)
    for bar, val in zip(bars, importancia.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', color=CLR_W, fontsize=9)
    ax.set_xlabel("Importância Relativa")
    ax.set_title("Variáveis mais influentes no modelo")
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()

    # Análise de threshold
    st.subheader("Análise de Threshold")
    th_rows = []
    for t in [0.40, 0.50, 0.60, 0.70, 0.80]:
        p_t = (proba_test > t).astype(int)
        th_rows.append({
            'Threshold':   f'{t:.0%}',
            'Precisão':    f"{precision_score(y_test, p_t, zero_division=0):.2%}",
            'Recall':      f"{recall_score(y_test, p_t):.2%}",
            'Selecionado': '✅' if t == THRESHOLD_FIXO else ''
        })
    st.dataframe(pd.DataFrame(th_rows).set_index('Threshold'), use_container_width=True)

    st.divider()

    # Heatmap de correlação
    st.subheader("Heatmap de Correlação entre Indicadores")
    corr_cols = ['INDE','IDA','IEG','IAA','IPS','IPP','IPV']
    corr = df_all[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG_DARK)
    ax.set_facecolor(BG_CARD)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
                annot_kws={"size": 9, "color": CLR_W},
                linewidths=0.5, linecolor='#1e3a52')
    ax.set_title("Correlação entre Indicadores × INDE", color=CLR_W, pad=12)
    plt.setp(ax.get_xticklabels(), color=CLR_W, rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), color=CLR_W, rotation=0)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

# =============================================================================
# ABA 4 — VÍDEO
# =============================================================================
with tab4:
    st.header("🎥 Vídeo de Apresentação do Datathon")
    st.markdown("Assista ao pitch do projeto detalhando a evolução dos indicadores da Passos Mágicos.")
    youtube_url = "https://youtu.be/Fqq_1ExsETw?si=XenMar6fN2v6cjbW"
    st.video(youtube_url)
