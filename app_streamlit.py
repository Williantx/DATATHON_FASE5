import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos - Predição de Risco Educacional",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar modelo e scaler
@st.cache_resource
def load_model_and_scaler():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'src', 'optimized_model.pkl')
    scaler_path = os.path.join(base_dir, 'src', 'scaler_v2.pkl')
    feature_cols_path = os.path.join(base_dir, 'src', 'feature_cols.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_cols_path)
    
    return model, scaler, feature_cols

# Função para fazer predição
def predict_risk(model, scaler, feature_cols, input_data):
    """Faz a predição de risco com base nos dados de entrada."""
    # Criar DataFrame com as features
    df_input = pd.DataFrame([input_data], columns=feature_cols)
    
    # Normalizar
    df_input_scaled = scaler.transform(df_input)
    
    # Predição
    prediction = model.predict(df_input_scaled)[0]
    prediction_proba = model.predict_proba(df_input_scaled)[0]
    
    return prediction, prediction_proba

# Interface principal
def main():
    # Carregar modelo
    model, scaler, feature_cols = load_model_and_scaler()
    
    # Título e descrição
    st.title("📚 Passos Mágicos - Sistema de Predição de Risco Educacional")
    st.markdown("""
    Este sistema utiliza **Machine Learning** para identificar alunos em risco de defasagem educacional, 
    permitindo intervenções proativas e personalizadas.
    """)
    
    # Sidebar para entrada de dados
    st.sidebar.header("📝 Dados do Aluno")
    
    # Inputs principais
    st.sidebar.subheader("Indicadores Principais")
    ida = st.sidebar.slider("IDA - Desempenho Acadêmico", 0.0, 10.0, 7.0, 0.1)
    ieg = st.sidebar.slider("IEG - Engajamento", 0.0, 10.0, 7.5, 0.1)
    ips = st.sidebar.slider("IPS - Aspectos Psicossociais", 0.0, 10.0, 6.5, 0.1)
    ipp = st.sidebar.slider("IPP - Aspectos Psicopedagógicos", 0.0, 10.0, 7.0, 0.1)
    iaa = st.sidebar.slider("IAA - Autoavaliação", 0.0, 10.0, 8.0, 0.1)
    ipv = st.sidebar.slider("IPV - Ponto de Virada", 0.0, 10.0, 7.5, 0.1)
    
    # Inputs adicionais
    st.sidebar.subheader("Informações Adicionais")
    ano = st.sidebar.selectbox("Ano", [2022, 2023, 2024], index=2)
    genero = st.sidebar.selectbox("Gênero", ["Feminino", "Masculino", "NAO_INFORMADO"])
    fase = st.sidebar.selectbox("Fase", ["ALFA", "FASE 1", "FASE 2", "FASE 3", "FASE 4", "FASE 5", "FASE 6", "FASE 7", "FASE 8", "NAO_INFORMADO"])
    
    # Encoding de variáveis categóricas (simplificado)
    genero_encoded = {"Feminino": 0, "Masculino": 1, "NAO_INFORMADO": 2}[genero]
    fase_encoded = {"ALFA": 0, "FASE 1": 1, "FASE 2": 2, "FASE 3": 3, "FASE 4": 4, "FASE 5": 5, "FASE 6": 6, "FASE 7": 7, "FASE 8": 8, "NAO_INFORMADO": 9}[fase]
    
    # Calcular features derivadas
    ida_ieg_interaction = ida * ieg
    ipp_ips_interaction = ipp * ips
    ida_ieg_ratio = ida / (ieg + 0.001)
    mean_indicators = np.mean([ida, ieg, ips, ipp, iaa, ipv])
    
    # Features temporais (valores padrão para novos alunos)
    ida_lag1 = st.sidebar.number_input("IDA (Ano Anterior)", 0.0, 10.0, ida, 0.1)
    ieg_lag1 = st.sidebar.number_input("IEG (Ano Anterior)", 0.0, 10.0, ieg, 0.1)
    ips_lag1 = st.sidebar.number_input("IPS (Ano Anterior)", 0.0, 10.0, ips, 0.1)
    
    ida_delta = ida - ida_lag1
    ieg_delta = ieg - ieg_lag1
    
    # Montar vetor de features
    input_data = [
        ida, ieg, ips, ipp, iaa, ipv,
        ida_ieg_interaction, ipp_ips_interaction, ida_ieg_ratio,
        mean_indicators, ida_lag1, ieg_lag1, ips_lag1,
        ida_delta, ieg_delta, genero_encoded, fase_encoded, ano
    ]
    
    # Botão de predição
    if st.sidebar.button("🔮 Prever Risco", type="primary"):
        # Fazer predição
        prediction, prediction_proba = predict_risk(model, scaler, feature_cols, input_data)
        
        # Exibir resultados
        st.header("📊 Resultado da Predição")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Classificação", "Alto Risco" if prediction == 1 else "Baixo Risco", 
                     delta="Atenção" if prediction == 1 else "OK", delta_color="inverse")
        
        with col2:
            st.metric("Probabilidade de Risco", f"{prediction_proba[1]*100:.1f}%")
        
        with col3:
            st.metric("Confiança", f"{max(prediction_proba)*100:.1f}%")
        
        # Gráfico de probabilidades
        st.subheader("📈 Distribuição de Probabilidades")
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ['Baixo Risco', 'Alto Risco']
        colors = ['#28a745', '#dc3545']
        ax.bar(labels, prediction_proba, color=colors, alpha=0.7)
        ax.set_ylabel('Probabilidade')
        ax.set_ylim(0, 1)
        ax.set_title('Probabilidade de Risco de Defasagem')
        for i, v in enumerate(prediction_proba):
            ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontweight='bold')
        st.pyplot(fig)
        
        # Recomendações
        st.subheader("💡 Recomendações")
        if prediction == 1:
            st.error("""
            **Aluno em Alto Risco de Defasagem**
            
            Recomendações:
            - 🎯 **Intervenção Imediata**: Agendar reunião com o aluno e responsáveis
            - 📚 **Reforço Acadêmico**: Oferecer aulas de reforço nas disciplinas com baixo desempenho
            - 🤝 **Mentoria**: Designar um mentor para acompanhamento próximo
            - 🧠 **Suporte Psicopedagógico**: Avaliar necessidade de apoio psicológico ou pedagógico
            - 📊 **Monitoramento Frequente**: Acompanhar evolução semanalmente
            """)
        else:
            st.success("""
            **Aluno com Baixo Risco de Defasagem**
            
            Recomendações:
            - ✅ **Manutenção**: Continuar com as atividades regulares do programa
            - 🌟 **Estímulo**: Incentivar participação em atividades de liderança
            - 📈 **Desafios**: Propor atividades mais desafiadoras para estimular crescimento
            - 👥 **Mentoria Reversa**: Considerar o aluno como mentor de colegas em risco
            """)
        
        # Análise de indicadores
        st.subheader("📋 Análise Detalhada dos Indicadores")
        
        indicators_df = pd.DataFrame({
            'Indicador': ['IDA', 'IEG', 'IPS', 'IPP', 'IAA', 'IPV'],
            'Valor': [ida, ieg, ips, ipp, iaa, ipv],
            'Status': [
                '✅ Adequado' if ida >= 7 else '⚠️ Atenção' if ida >= 5 else '❌ Crítico',
                '✅ Adequado' if ieg >= 7 else '⚠️ Atenção' if ieg >= 5 else '❌ Crítico',
                '✅ Adequado' if ips >= 7 else '⚠️ Atenção' if ips >= 5 else '❌ Crítico',
                '✅ Adequado' if ipp >= 7 else '⚠️ Atenção' if ipp >= 5 else '❌ Crítico',
                '✅ Adequado' if iaa >= 7 else '⚠️ Atenção' if iaa >= 5 else '❌ Crítico',
                '✅ Adequado' if ipv >= 7 else '⚠️ Atenção' if ipv >= 5 else '❌ Crítico'
            ]
        })
        
        st.dataframe(indicators_df, use_container_width=True)
    
    # Informações sobre o modelo
    with st.expander("ℹ️ Sobre o Modelo"):
        st.markdown("""
        ### Modelo Preditivo de Risco Educacional
        
        **Algoritmo**: Gradient Boosting Classifier (Otimizado)
        
        **Performance**:
        - ROC-AUC Score: 0.7020
        - Acurácia: 70%
        
        **Features Mais Importantes**:
        1. Razão IDA/IEG (Desempenho vs Engajamento)
        2. Ano
        3. Fase
        4. IDA (Desempenho Acadêmico)
        5. IEG (Engajamento)
        
        **Interpretação**:
        O modelo utiliza uma combinação de indicadores acadêmicos, psicossociais e psicopedagógicos 
        para identificar padrões que antecedem quedas de desempenho ou aumento da defasagem.
        """)

if __name__ == "__main__":
    main()
