# Obesity Risk Stratification System (DSS)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-red)
![Scikit-learn](https://img.shields.io/badge/ML-RandomForest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Sobre o Projeto

Este projeto consiste em um **Sistema de Apoio à Decisão Clínica (CDSS)** voltado para a triagem e predição de níveis de obesidade. Utilizando algoritmos de Machine Learning treinados em dados antropométricos e comportamentais, a ferramenta oferece uma interface para profissionais de saúde estimarem riscos e analisarem padrões populacionais.

O sistema foi desenvolvido como parte do **Tech Challenge - Fase 4 (Data Analytics)**, demonstrando o ciclo completo de ciência de dados: desde a análise exploratória e engenharia de atributos até o deploy do modelo em produção.

### 🎯 Objetivos
- **Triagem Preditiva:** Classificação automática em 7 níveis de peso (do baixo peso à obesidade mórbida) com base em 16 variáveis.
- **Inteligência Clínica:** Geração de laudos automáticos e insights sobre hábitos de risco (sedentarismo, dieta, hereditariedade).
- **Gestão de Saúde:** Painel analítico para visualização de tendências epidemiológicas.

---

## ⚙️ Arquitetura e Tecnologia

O projeto segue uma arquitetura modular focada em reprodutibilidade:

* **Linguagem:** Python 3.11+
* **Frontend/Dashboard:** Streamlit
* **Machine Learning:** Scikit-learn (Random Forest Classifier)
* **Processamento de Dados:** Pandas & Numpy
* **Visualização:** Matplotlib (Customizado para relatórios clínicos)
* **Persistência:** Joblib

**Performance do Modelo:**
* **Algoritmo:** Random Forest (Otimizado via GridSearch/Validação Cruzada)
* **Acurácia Global:** ~97.8%
* **Features:** 17 atributos (incluindo cálculo automático de IMC e tratamento de variáveis categóricas).

---

## 📂 Estrutura do Repositório

```text
/
├── app/
│   └── app.py              # Aplicação principal (Frontend Streamlit)
│
├── database/
│   └── Obesity.csv         # Dataset original (UCI Repository)
│
├── models/
│   ├── obesity_pipeline.joblib  # Pipeline treinado (Pré-processamento + Modelo)
│   └── feature_columns.json     # Metadados das colunas para inferência
│
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA, Feature Engineering e Treino
│
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação
