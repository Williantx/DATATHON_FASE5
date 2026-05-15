📊 Base de Dados

O projeto utilizou uma base longitudinal contendo:

3.030 registros
Dados entre 2022 e 2024
Mais de 34 variáveis multidimensionais

As variáveis incluem:

Indicadores acadêmicos
Engajamento
Indicadores psicossociais
Evolução pedagógica
Informações comportamentais

🔍 Principais Indicadores

Indicador	Descrição
IAN	Indicador de Adequação de Nível
IDA	Indicador de Desempenho Acadêmico
IEG	Indicador de Engajamento
IAA	Indicador de Autoavaliação
IPS	Indicador Psicossocial
IPP	Indicador Psicopedagógico
IPV	Indicador de Ponto de Virada
INDE	Índice de Desenvolvimento Educacional
📈 Principais Insights
✅ Evolução Temporal

A taxa de defasagem reduziu de:
69,9% (2022)
para 46,2% (2024)
O INDE médio apresentou evolução ao longo do tempo.
✅ Engajamento como fator-chave

O projeto identificou forte relação entre:

Engajamento (IEG)
Desempenho acadêmico (IDA)
Evolução pedagógica
Correlações observadas:
Relação	Correlação
IDA × INDE	0.78
IEG × INDE	0.74
✅ Evolução Pedagógica

O nível Topázio evoluiu de:

15,1% em 2022
para 30,9% em 2024

Indicando avanço consistente na trajetória dos alunos.

🤖 Machine Learning
📌 Objetivo

Desenvolver um modelo capaz de identificar alunos em risco de defasagem educacional.

🔧 Pipeline do Modelo

Tratamento de dados
Feature Engineering
Balanceamento com SMOTE
Random Forest
Ajuste de Threshold
Avaliação de performance

⚙️ Variáveis Utilizadas

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
📊 Resultados do Modelo
Métrica	Resultado
ROC-AUC	0.75
Precisão	86%
Recall	20%
Accuracy	73%


🧠 Estratégia do Modelo

O modelo foi calibrado para priorizar precisão e reduzir falsos positivos.

Essa abordagem aumenta a confiabilidade pedagógica dos alertas gerados.

📌 Principais Variáveis do Modelo

As variáveis mais importantes identificadas foram:

Fase no programa
Engajamento (IEG)
Indicadores psicossociais
Desempenho acadêmico
💻 Aplicação Prática

Foi desenvolvida uma aplicação utilizando Streamlit para:

Monitoramento de indicadores
Visualização de riscos
Apoio à decisão pedagógica
Acompanhamento preventivo
🌍 Impacto Social

O projeto demonstra como Data Analytics pode ser utilizado para:

Antecipar riscos educacionais
Melhorar intervenções pedagógicas
Apoiar decisões humanas com inteligência
Reduzir defasagem e evasão escolar
🛠️ Tecnologias Utilizadas
Linguagens e Frameworks
Python
Streamlit
Bibliotecas
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Imbalanced-Learn
🚀 Como Executar o Projeto
1. Clonar o repositório
git clone https://github.com/Williantx/DATATHON_FASE5_Passos_Magicos
2. Instalar dependências
pip install -r requirements.txt
3. Executar Streamlit
streamlit run app.py
📚 Estrutura Analítica

O projeto foi dividido em:

Tratamento e preparação dos dados
Análise exploratória
Storytelling analítico
Modelagem preditiva
Aplicação prática
👨‍🎓 Projeto Acadêmico

Projeto desenvolvido para a Pós-Graduação em Data Analytics — FIAP.

📌 Conclusão

O projeto demonstrou que a defasagem educacional é um fenômeno multifatorial e que o uso de dados permite antecipar riscos e apoiar intervenções pedagógicas mais assertivas.

Mais do que prever problemas, o objetivo foi criar oportunidades antes que a defasagem se consolide.

✨ Frase Final

“Dados também podem transformar vidas.”
