# 📊 Análise de Dados e Predição: Passos Mágicos

Projeto desenvolvido para a **Pós-Graduação em Data Analytics — FIAP**, com o objetivo de combater a defasagem educacional através de dados e inteligência artificial.

## 📁 Base de Dados
O projeto utilizou uma base longitudinal composta por:
* **3.030** registros.
* Dados coletados entre **2022 e 2024**.
* Mais de **34 variáveis** multidimensionais, incluindo:
    * Indicadores acadêmicos e pedagógicos.
    * Engajamento e indicadores psicossociais.
    * Informações comportamentais.

## 🔍 Principais Indicadores
| Indicador | Descrição |
| :--- | :--- |
| **IAN** | Indicador de Adequação de Nível |
| **IDA** | Indicador de Desempenho Acadêmico |
| **IEG** | Indicador de Engajamento |
| **IAA** | Indicador de Autoavaliação |
| **IPS** | Indicador Psicossocial |
| **IPP** | Indicador Psicopedagógico |
| **IPV** | Indicador de Ponto de Virada |
| **INDE** | Índice de Desenvolvimento Educacional |

## 📈 Principais Insights
### ✅ Evolução Temporal
* **Redução da Defasagem:** De **69,9% (2022)** para **46,2% (2024)**.
* **Avanço Pedagógico:** O nível **Topázio** cresceu de 15,1% para 30,9% no período.
* O **INDE médio** apresentou evolução constante ao longo do tempo.

### ✅ Fatores-Chave de Sucesso
Identificou-se uma forte correlação entre Engajamento (**IEG**), Desempenho (**IDA**) e o Índice de Desenvolvimento (**INDE**).
* **IDA × INDE:** 0.78
* **IEG × INDE:** 0.74

## 🤖 Machine Learning
### 📌 Objetivo
Desenvolver um modelo preditivo para identificar alunos em risco de defasagem educacional.

### 🔧 Pipeline do Modelo
1. Tratamento de dados & Feature Engineering
2. Balanceamento de classes com **SMOTE**
3. Algoritmo **Random Forest**
4. Ajuste de Threshold para confiabilidade pedagógica
5. Avaliação de performance

### 📊 Resultados do Modelo
| Métrica | Resultado |
| :--- | :--- |
| **ROC-AUC** | 0.75 |
| **Precisão** | 86% |
| **Recall** | 20% |
| **Acurácia** | 73% |

> **Estratégia:** O modelo prioriza a **Precisão** para reduzir falsos positivos, garantindo que os alertas gerados para a equipe pedagógica sejam altamente confiáveis.

### ⚙️ Variáveis (Features) Utilizadas
```python
features = [
    'IDA', 'IEG', 'IAA', 'IPS', 
    'IPP', 'Fase_Num', 'IPV', 'Anos_No_Programa'
]
