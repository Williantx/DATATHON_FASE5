# 📊 Análise de Dados e Predição: Passos Mágicos

Projeto desenvolvido para a **Pós-Graduação em Data Analytics — FIAP**, com o objetivo de combater a defasagem educacional através de dados e inteligência artificial.

## 🔗 Links do Projeto
* **🚀 Aplicação Streamlit:** [Analytics - Passos Mágicos](https://passosmagicos.streamlit.app/)
* **🎥 Vídeo de Demonstração:** [Assista aqui](https://drive.google.com/file/d/1QyP_YfHQeIK6cABuL_463aTfFvuyQXQU/view?usp=sharing)
* **🖼️ Apresentação Completa:** [Acesse os slides](https://docs.google.com/presentation/d/19KuBSyKADQBgzwnsW4526j6pSOXu2c8D/edit?usp=drive_link&ouid=113160887725821989157&rtpof=true&sd=true)

## 📂 Estrutura do Repositório
A organização do projeto segue a seguinte hierarquia de pastas:

* **`Arquivos/`**: Bases de dados brutas e processadas.
* **`Streamlit/`**: Código-fonte da aplicação web e dashboards.
* **`Storytelling/`**: Notebooks de análise exploratória e apresentações.
* **`README.md`**: Documentação principal do projeto.

## 📁 Base de Dados
O projeto utilizou uma base longitudinal contendo:
* **3.030** registros de alunos.
* Dados históricos entre **2022 e 2024**.
* Mais de **34 variáveis** multidimensionais (acadêmicas, psicossociais e comportamentais).

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
* **Redução da Defasagem:** A taxa caiu de **69,9% (2022)** para **46,2% (2024)**.
* **Avanço Pedagógico:** O nível **Topázio** evoluiu de 15,1% para 30,9% no período, indicando melhora na trajetória dos alunos.

### ✅ Fatores-Chave
Forte correlação identificada entre Engajamento e Desempenho:
* **IDA × INDE:** 0.78
* **IEG × INDE:** 0.74

## 🤖 Machine Learning
### 📌 Objetivo
Identificar preventivamente alunos com risco de defasagem educacional.

### 📊 Resultados do Modelo (Random Forest)
| Métrica | Resultado |
| :--- | :--- |
| **ROC-AUC** | 0.96 |
| **Precisão** | 93,24% |
| **Recall** | 69,42% |
| **Accuracy** | 083 % |

> **Estratégia:** O modelo foi calibrado com foco em **Precisão** para garantir que as intervenções pedagógicas sejam direcionadas com alta assertividade, evitando falsos alertas.

### ⚙️ Features Utilizadas
```python
features = [
    'IDA', 'IEG', 'IAA', 'IPS', 
    'IPP', 'Fase_Num', 'IPV', 'Anos_No_Programa'
]
