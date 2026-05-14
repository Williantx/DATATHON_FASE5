import streamlit as st
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
