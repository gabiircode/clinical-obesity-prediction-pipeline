# Obesity Risk Stratification System (DSS)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-red)
![Scikit-learn](https://img.shields.io/badge/ML-RandomForest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Sobre o Projeto

Este projeto consiste em um **Sistema de Apoio à Decisão Clínica (CDSS)** voltado para a triagem e predição de níveis de obesidade. A ferramenta utiliza algoritmos de Machine Learning para processar dados antropométricos e comportamentais, oferecendo uma interface intuitiva para profissionais de saúde estimarem riscos e analisarem padrões populacionais.

O projeto foi desenvolvido como entrega final do **Tech Challenge - Fase 4 (Data Analytics)**, englobando todo o ciclo de vida dos dados: Análise Exploratória (EDA), Pré-processamento, Treinamento de Modelo e Deploy em Produção.

🔗 **Link da Aplicação:** [INSIRA SEU LINK DO STREAMLIT AQUI]

---

## 🧠 Principais Insights do Estudo

Durante a análise exploratória dos dados (disponível na pasta `notebooks`), foram identificados padrões comportamentais críticos que fundamentam as predições do modelo:

1.  **O Peso da Genética:** O histórico familiar apresentou correlação extremamente forte com a obesidade (Graus I, II e III). Indivíduos com parentes obesos têm probabilidade significativamente maior de desenvolver a condição, independente de alguns fatores ambientais.
2.  **Transporte Ativo vs. Passivo:** O uso de transporte público e caminhada mostrou-se um fator protetor, enquanto o uso frequente de automóveis está fortemente associado a níveis mais altos de IMC.
3.  **O Perigo das "Beliscadas":** A variável `CAEC` (Comer entre refeições) demonstrou alto poder preditivo. Pacientes que relataram "comer frequentemente" entre as refeições principais tendem a migrar para as faixas de Sobrepeso e Obesidade.
4.  **Hidratação:** O baixo consumo de água foi um traço comum nos grupos de maior risco, sugerindo que a hidratação pode ser um marcador indireto de consciência alimentar.

---

## 📂 Estrutura do Repositório

O projeto segue uma arquitetura modular para garantir escalabilidade e reprodutibilidade:

```text
/
├── app/
│   └── app.py                        # Frontend da aplicação (Streamlit)
│
├── database/
│   └── Obesity.csv                   # Dataset bruto (UCI Machine Learning Repository)
│
├── models/
│   ├── obesity_pipeline.joblib       # Pipeline serializado (Scaler + Encoder + Modelo)
│   └── feature_columns.json          # Metadados para garantir a ordem das features
│
├── notebooks/
│   └── 01_exploratory_analysis_train.ipynb  # Estudo detalhado, gráficos e treinamento
│
├── outputs/
│   └── metrics.json                  # Relatório de performance (Acurácia, F1-Score)
│
├── requirements.txt                  # Dependências do ambiente
└── README.md                         # Documentação oficial
