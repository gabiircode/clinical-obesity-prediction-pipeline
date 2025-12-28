import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Obesity Predictor", layout="wide")

# ---------------------------------------------------------
# Paths (funcionam rodando: python -m streamlit run app/app.py)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../techchallenge_obesity/app
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))     # .../techchallenge_obesity

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "obesity_pipeline.joblib")
COLS_PATH = os.path.join(PROJECT_DIR, "models", "feature_columns.json")
METRICS_PATH = os.path.join(PROJECT_DIR, "outputs", "metrics.json")
DATA_PATH = os.path.join(PROJECT_DIR, "database", "Obesity.csv")


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_feature_columns():
    with open(COLS_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        # padroniza TER -> TUE se necessário
        if "TER" in df.columns and "TUE" not in df.columns:
            df = df.rename(columns={"TER": "TUE"})
        return df
    return None


# ---------------------------------------------------------
# Validations
# ---------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Modelo não encontrado em: {MODEL_PATH}")
    st.stop()

if not os.path.exists(COLS_PATH):
    st.error(f"❌ feature_columns.json não encontrado em: {COLS_PATH}")
    st.stop()

clf = load_model()
feature_cols = load_feature_columns()
metrics = load_metrics()
df_full = load_dataset()

# tenta detectar target se dataset existir
TARGET_COL = None
if df_full is not None:
    possible_targets = ["NObeyesdad", "Obesity", "Obesity_level", "obesity_level"]
    for c in possible_targets:
        if c in df_full.columns:
            TARGET_COL = c
            break
    if TARGET_COL is None:
        TARGET_COL = df_full.columns[-1]


# ---------------------------------------------------------
# UI - Header
# ---------------------------------------------------------
st.title("Sistema Preditivo de Obesidade")
st.write("Aplicação em **Streamlit** para apoiar **triagem** do **nível de obesidade** com base em hábitos e medidas do indivíduo.")

if metrics and "accuracy" in metrics:
    st.success(f"✅ Acurácia do modelo no conjunto de teste: **{metrics['accuracy']*100:.2f}%**")

tabs = st.tabs(["🔮 Predição", "📊 Painel Analítico", "ℹ️ Sobre"])


# =========================================================
# Helpers (labels + tipos)
# =========================================================

# Siglas -> perguntas (experiência melhor)
QUESTION_LABELS = {
    "Gender": "Gênero",
    "Age": "Idade (anos)",
    "Height": "Altura (m)",
    "Weight": "Peso (kg)",
    "family_history": "Algum familiar tem/teve excesso de peso?",
    "FAVC": "Você consome alimentos altamente calóricos com frequência?",
    "FCVC": "Você costuma comer vegetais nas suas refeições? (1=baixo, 3=alto)",
    "NCP": "Quantas refeições principais você faz por dia?",
    "CAEC": "Você costuma comer entre as refeições?",
    "SMOKE": "Você fuma?",
    "CH2O": "Quanta água você bebe por dia? (1=baixo, 3=alto)",
    "SCC": "Você monitora as calorias ingeridas diariamente?",
    "FAF": "Com que frequência pratica atividade física? (dias/semana)",
    "TUE": "Tempo de uso de tecnologia (0=pouco, 2=alto)",
    "TER": "Tempo de uso de tecnologia (0=pouco, 2=alto)",
    "CALC": "Com que frequência você consome álcool?",
    "MTRANS": "Qual meio de transporte você mais utiliza?"
}

# Campos que precisam ser inteiros (escala, segundo dicionário/ruído)
INT_SCALE_COLS = {"FCVC", "NCP", "CH2O", "FAF", "TUE"}

# Dicionário das classes (para explicar no app)
CLASS_DESCRIPTION = {
    "Insufficient_Weight": "Peso abaixo do recomendado para a altura (baixo peso).",
    "Normal_Weight": "Peso adequado de acordo com altura e idade.",
    "Overweight_Level_I": "Sobrepeso nível I (acima do peso ideal).",
    "Overweight_Level_II": "Sobrepeso nível II (limite entre sobrepeso e obesidade).",
    "Obesity_Type_I": "Obesidade grau I.",
    "Obesity_Type_II": "Obesidade grau II.",
    "Obesity_Type_III": "Obesidade grau III (obesidade severa)."
}

CLINICAL_GUIDANCE = {
    "Insufficient_Weight": "Avaliar risco nutricional, investigar causas e acompanhar com profissional de saúde.",
    "Normal_Weight": "Manter hábitos saudáveis e atividade física regular. Acompanhamento preventivo.",
    "Overweight_Level_I": "Reforçar hábitos alimentares saudáveis e aumentar atividade física.",
    "Overweight_Level_II": "Recomendável avaliação nutricional e acompanhamento para evitar progressão.",
    "Obesity_Type_I": "Acompanhamento médico e nutricional. Plano de mudança de estilo de vida.",
    "Obesity_Type_II": "Intervenção multidisciplinar recomendada (médico, nutrição, atividade física).",
    "Obesity_Type_III": "Acompanhamento médico intensivo. Avaliar comorbidades e plano terapêutico."
}


def nice_label(col_name: str) -> str:
    """Retorna label amigável (sem sigla)"""
    return QUESTION_LABELS.get(col_name, col_name)


def coerce_int(v, fallback=0) -> int:
    """Converte pra int com segurança (round antes, por causa do ruído)"""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return int(fallback)
        return int(round(float(v)))
    except Exception:
        return int(fallback)


def safe_number_input(label, value, mn=None, mx=None, step=None, as_int=False):
    """
    Evita StreamlitMixedNumericTypesError:
    - Se as_int=True: min/max/value/step TODOS int
    - Caso contrário: TODOS float
    """
    if as_int:
        v = coerce_int(value, fallback=0)
        kwargs = {
            "value": int(v),
            "step": int(step) if step is not None else 1,
        }
        if mn is not None: kwargs["min_value"] = int(mn)
        if mx is not None: kwargs["max_value"] = int(mx)
        return st.number_input(label, **kwargs)
    else:
        v = float(value) if value is not None else 0.0
        kwargs = {
            "value": float(v),
            "step": float(step) if step is not None else 0.1,
        }
        if mn is not None: kwargs["min_value"] = float(mn)
        if mx is not None: kwargs["max_value"] = float(mx)
        return st.number_input(label, **kwargs)


def compute_bmi(height_m, weight_kg):
    try:
        h = float(height_m)
        w = float(weight_kg)
        if h <= 0:
            return None
        return w / (h ** 2)
    except Exception:
        return None


# =========================================================
# TAB 1: PREDIÇÃO (CORRIGIDA E BLINDADA)
# =========================================================
with tabs[0]:
    st.markdown("### 📋 Formulário de Triagem")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 20px; color: #31333F;'>
        Preencha os dados do paciente para realizar a predição. 
        O sistema aceita altura com ponto ou vírgula (ex: 1.70 ou 1,70).
        </div>
        """, 
        unsafe_allow_html=True
    )

    with st.form("triagem_form"):
        
        # --- CARD 1: PERFIL ---
        with st.container(border=True):
            st.markdown("#### 👤 Perfil do Paciente")
            c1, c2 = st.columns(2)
            
            with c1:
                gender_opt = ["Masculino", "Feminino"]
                gender_val = st.selectbox("Gênero", gender_opt)
                age_val = st.number_input("Idade (anos)", min_value=10, max_value=100, value=25)
                fam_val = st.selectbox("Histórico familiar de sobrepeso?", ["Não", "Sim"])

            with c2:
                # Text Input para permitir virgula, mas com validação forte depois
                height_txt = st.text_input("Altura (m)", value="1.70", placeholder="Ex: 1.70 ou 1,70")
                weight_val = st.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1, format="%.1f")

        # --- CARD 2: ALIMENTAÇÃO ---
        with st.container(border=True):
            st.markdown("#### 🥗 Hábitos Alimentares")
            col_food1, col_food2 = st.columns(2)
            
            with col_food1:
                fcvc_val = st.slider("Consumo de vegetais (FCVC)", 1, 3, 2, help="1=Nunca, 2=Às vezes, 3=Sempre")
                ncp_val = st.number_input("Refeições principais (dia)", 1, 6, 3)
                scc_val = st.selectbox("Monitora calorias?", ["Não", "Sim"])

            with col_food2:
                # Slider de texto mapeado
                ch2o_val = st.select_slider("Água (CH2O)", options=["Baixo (<1L)", "Médio (1-2L)", "Alto (>2L)"], value="Médio (1-2L)")
                map_water_reverse = {"Baixo (<1L)": 1, "Médio (1-2L)": 2, "Alto (>2L)": 3}
                
                favc_val = st.selectbox("Alimentos calóricos frequentes?", ["Não", "Sim"])
                
                caec_opt = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
                caec_label = st.selectbox("Come entre refeições?", list(caec_opt.keys()), index=1)

        # --- CARD 3: ESTILO DE VIDA ---
        with st.container(border=True):
            st.markdown("#### 🏃 Estilo de Vida")
            c_life1, c_life2 = st.columns(2)
            
            with c_life1:
                smoke_val = st.selectbox("Fumante?", ["Não", "Sim"])
                
                calc_opt = {"Não": "no", "Às vezes": "Sometimes", "Frequente": "Frequently", "Sempre": "Always"}
                calc_label = st.selectbox("Álcool?", list(calc_opt.keys()), index=1)
                
                mtrans_opt = ["Transporte Público", "Automóvel", "Caminhada", "Motocicleta", "Bicicleta"]
                mtrans_val = st.selectbox("Transporte principal?", mtrans_opt)

            with c_life2:
                faf_val = st.slider("Atividade Física (dias/sem)", 0, 3, 1, help="0=Sedentário, 1=1-2 dias, 2=3-4 dias, 3=5+ dias")
                tue_val = st.slider("Tempo de Tela (h/dia)", 0, 2, 1, help="0=0-2h, 1=3-5h, 2=+5h")

        st.markdown("---")
        submitted = st.form_submit_button("🩺 Realizar Predição Clínica", type="primary", use_container_width=True)

    # --- LÓGICA DE PREDIÇÃO ---
    if submitted:
        # 1. Tratamento Robusto da Altura
        try:
            # Troca vírgula por ponto e remove espaços
            h_str = height_txt.replace(",", ".").strip()
            h = float(h_str)
            
            # Correção inteligente: se altura > 3, assume que digitou em cm (ex: 170)
            if h > 3:
                h = h / 100
                st.toast(f"Altura convertida automaticamente para {h:.2f}m", icon="📏")
            
            if h <= 0:
                st.error("Altura deve ser maior que zero.")
                st.stop()
                
        except ValueError:
            st.error("⚠️ Valor de altura inválido. Digite apenas números (ex: 1.75)")
            st.stop()

        # 2. Cálculo do IMC
        w = float(weight_val)
        bmi = w / (h ** 2)

        # 3. Mapeamentos para o Modelo
        map_yes_no = {"Sim": "yes", "Não": "no"}
        map_gender = {"Masculino": "Male", "Feminino": "Female"}
        
        map_trans = {
            "Transporte Público": "Public_Transportation",
            "Automóvel": "Automobile",
            "Caminhada": "Walking",
            "Motocicleta": "Motorbike",
            "Bicicleta": "Bike"
        }

        # Dicionário Input
        input_dict = {
            "Gender": map_gender[gender_val],
            "Age": age_val,
            "Height": h,
            "Weight": w,
            "family_history": map_yes_no[fam_val],
            "FAVC": map_yes_no[favc_val],
            "FCVC": fcvc_val,
            "NCP": ncp_val,
            "CAEC": caec_opt[caec_label],
            "SMOKE": map_yes_no[smoke_val],
            "CH2O": map_water_reverse[ch2o_val],
            "SCC": map_yes_no[scc_val], # Monitoramento calorias
            "FAF": faf_val,
            "TUE": tue_val,
            "CALC": calc_opt[calc_label],
            "MTRANS": map_trans.get(mtrans_val, "Public_Transportation"),
            "BMI": bmi
        }

        # 4. Predição
        try:
            # Garante ordem das colunas
            df_final = pd.DataFrame([input_dict]).reindex(columns=feature_cols)
            
            # Prediz
            pred = clf.predict(df_final)[0]
            
            # Tradução do resultado
            map_target_show = {
                "Insufficient_Weight": "Abaixo do Peso",
                "Normal_Weight": "Peso Normal",
                "Overweight_Level_I": "Sobrepeso Nível I",
                "Overweight_Level_II": "Sobrepeso Nível II",
                "Obesity_Type_I": "Obesidade Grau I",
                "Obesity_Type_II": "Obesidade Grau II",
                "Obesity_Type_III": "Obesidade Grau III"
            }
            pred_pt = map_target_show.get(pred, pred)
            
            desc = CLASS_DESCRIPTION.get(pred, "")
            guidance = CLINICAL_GUIDANCE.get(pred, "")

            # Definição de Cor do Resultado
            if "Obesity" in pred:
                status_color = "🔴"
            elif "Overweight" in pred:
                status_color = "🟡"
            elif "Insufficient" in pred:
                status_color = "🟠"
            else:
                status_color = "🟢"

            # 5. Exibição do Resultado
            st.write("")
            with st.container(border=True):
                st.markdown(f"### {status_color} Resultado: **{pred_pt}**")
                
                c_res1, c_res2 = st.columns([1, 2])
                with c_res1:
                    st.metric("IMC Calculado", f"{bmi:.2f} kg/m²")
                with c_res2:
                    st.info(f"**Interpretação:** {desc}")
                
                st.success(f"**Conduta Sugerida:** {guidance}")

        except Exception as e:
            st.error(f"Erro no processamento do modelo: {e}")

# =========================================================
# TAB 2: PAINEL CLÍNICO (100% PORTUGUÊS & ORDENADO)
# =========================================================
with tabs[1]:
    st.subheader("Análise Clínica e Estratificação de Risco")
    st.write("") 

    if df_full is None:
        st.error("Dataset indisponível.")
    else:
        # --- 1. PREPARAÇÃO, TRADUÇÃO E LIMPEZA ---
        dfp = df_full.copy()
        
        # 1.1 Tradução das Colunas de Texto (Sim/Não/Gênero)
        cols_translate = ["family_history", "FAVC", "SMOKE", "SCC"]
        for c in cols_translate:
            if c in dfp.columns:
                dfp[c] = dfp[c].map({"yes": "Sim", "no": "Não"}).fillna(dfp[c])
        
        if "Gender" in dfp.columns:
            dfp["Gender"] = dfp["Gender"].map({"Male": "Masculino", "Female": "Feminino"}).fillna(dfp["Gender"])

        # 1.2 TRADUÇÃO DAS CLASSES DE OBESIDADE (TARGET)
        # Isso garante que o filtro e os gráficos apareçam em Português
        map_target = {
            "Insufficient_Weight": "Abaixo do Peso",
            "Normal_Weight": "Peso Normal",
            "Overweight_Level_I": "Sobrepeso Nível I",
            "Overweight_Level_II": "Sobrepeso Nível II",
            "Obesity_Type_I": "Obesidade Grau I",
            "Obesity_Type_II": "Obesidade Grau II",
            "Obesity_Type_III": "Obesidade Grau III (Mórbida)"
        }
        dfp[TARGET_COL] = dfp[TARGET_COL].map(map_target).fillna(dfp[TARGET_COL])

        # 1.3 Garante cálculo IMC
        if "Height" in dfp.columns and "Weight" in dfp.columns:
            dfp["BMI"] = dfp["Weight"] / (dfp["Height"] ** 2)
        else:
            dfp["BMI"] = np.nan

        # 1.4 Converte escalas para inteiro
        for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
            if col in dfp.columns:
                dfp[col] = pd.to_numeric(dfp[col], errors='coerce').fillna(0).round().astype(int)

        # Médias Populacionais
        pop_bmi = dfp["BMI"].mean()
        pop_age = dfp["Age"].mean()

        # --- 2. MAPEAMENTO SEMÂNTICO (CORES E ORDEM) ---
        dict_faf = {
            "labels": {0: "Sedentário", 1: "1-2 dias/sem", 2: "3-4 dias/sem", 3: "5+ dias/sem"},
            "order": ["Sedentário", "1-2 dias/sem", "3-4 dias/sem", "5+ dias/sem"],
            "colors": {"Sedentário": "#d62728", "1-2 dias/sem": "#ff7f0e", "3-4 dias/sem": "#bcbd22", "5+ dias/sem": "#2ca02c"}
        }

        dict_tue = {
            "labels": {0: "Baixo (0-2h)", 1: "Médio (3-5h)", 2: "Alto (+5h)"},
            "order": ["Baixo (0-2h)", "Médio (3-5h)", "Alto (+5h)"],
            "colors": {"Baixo (0-2h)": "#2ca02c", "Médio (3-5h)": "#ff7f0e", "Alto (+5h)": "#d62728"}
        }

        dict_water = {
            "labels": {1: "< 1 Litro", 2: "1-2 Litros", 3: "> 2 Litros"},
            "order": ["< 1 Litro", "1-2 Litros", "> 2 Litros"],
            "colors": {"< 1 Litro": "#d62728", "1-2 Litros": "#1f77b4", "> 2 Litros": "#2ca02c"}
        }

        dict_veg = {
            "labels": {1: "Nunca", 2: "Às vezes", 3: "Sempre"},
            "order": ["Nunca", "Às vezes", "Sempre"],
            "colors": {"Nunca": "#d62728", "Às vezes": "#ff7f0e", "Sempre": "#2ca02c"}
        }

        dict_cal = {
            "labels": {"Sim": "Sim", "Não": "Não"},
            "order": ["Sim", "Não"],
            "colors": {"Sim": "#d62728", "Não": "#2ca02c"}
        }

        # --- 3. FILTRO COM ORDENAÇÃO CLÍNICA ---
        # Define a ordem lógica (não alfabética) para o Selectbox
        ordem_clinica = [
            "Abaixo do Peso", 
            "Peso Normal", 
            "Sobrepeso Nível I", 
            "Sobrepeso Nível II", 
            "Obesidade Grau I", 
            "Obesidade Grau II", 
            "Obesidade Grau III (Mórbida)"
        ]
        
        # Filtra apenas as classes que realmente existem no dataset carregado
        classes_existentes = dfp[TARGET_COL].unique().tolist()
        lista_final = ["Visão Geral"] + [c for c in ordem_clinica if c in classes_existentes]
        
        selected = st.selectbox("Selecione o Perfil Clínico para Análise:", lista_final)
        st.write("")

        if selected == "Visão Geral":
            df_sel = dfp
            is_general = True
        else:
            df_sel = dfp[dfp[TARGET_COL] == selected]
            is_general = False

        st.divider()

        # --- 4. KPIs PADRONIZADOS ---
        k1, k2, k3, k4 = st.columns(4)
        
        curr_bmi = df_sel["BMI"].mean()
        curr_age = df_sel["Age"].mean()
        curr_fam = (df_sel["family_history"] == "Sim").mean() * 100 
        
        delta_bmi = (curr_bmi - pop_bmi) if not is_general else 0
        delta_age = (curr_age - pop_age) if not is_general else 0

        k1.metric("Idade Média", f"{curr_age:.1f} anos", delta=f"{delta_age:.1f} vs Geral", delta_color="off")
        k2.metric("IMC Médio", f"{curr_bmi:.1f} kg/m²", delta=f"{delta_bmi:.1f} vs Geral", delta_color="inverse")
        k3.metric("Histórico Familiar", f"{curr_fam:.0f}%", help="% com antecedentes familiares")
        k4.metric("Nº Pacientes", f"{len(df_sel)}")
        
        st.write("")

        # --- 5. FUNÇÃO DE PLOTAGEM ---
        def plot_clinical_bar(data, col, meta_dict, title, insight_text):
            if col not in data.columns: return
            
            s_data = data[col].copy()
            mapped = s_data.map(meta_dict["labels"]).fillna("Outros")
            
            counts = mapped.value_counts()
            df_counts = pd.DataFrame(counts).reindex(meta_dict["order"]).fillna(0)
            df_counts.columns = ["Qtd"]
            
            if df_counts["Qtd"].sum() == 0: return

            bar_colors = [meta_dict["colors"].get(label, "#cccccc") for label in df_counts.index]

            fig, ax = plt.subplots(figsize=(5, 2.2)) 
            bars = ax.barh(df_counts.index, df_counts["Qtd"], color=bar_colors)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            
            ax.set_title(title, fontsize=10, fontweight='bold', pad=10, color='#333333', loc='left')
            ax.bar_label(bars, fmt='%d', padding=5, fontsize=9, fontweight='bold')
            
            ax.get_yaxis().set_visible(True)
            ax.tick_params(axis='y', length=0, labelsize=9)
            
            st.pyplot(fig)
            st.info(f"🩺 {insight_text}")

        # --- 6. GRÁFICOS ---
        st.subheader("🏃 Estilo de Vida e Sedentarismo")
        c1, c2 = st.columns(2)
        
        with c1:
            pct_sedentary = (df_sel["FAF"] == 0).mean()
            txt_faf = f"Alerta: {pct_sedentary:.0%} do grupo é sedentário." if pct_sedentary > 0.5 else "Nível de atividade física razoável."
            plot_clinical_bar(df_sel, "FAF", dict_faf, "Freq. Atividade Física", txt_faf)
            
        with c2:
            pct_high_screen = (df_sel["TUE"] == 2).mean()
            txt_tue = f"Atenção: {pct_high_screen:.0%} passam +5h em telas." if pct_high_screen > 0.4 else "Tempo de tela controlado."
            plot_clinical_bar(df_sel, "TUE", dict_tue, "Tempo de Tecnologia", txt_tue)

        st.divider()

        st.subheader("🥦 Perfil Nutricional")
        c3, c4, c5 = st.columns(3)
        
        with c3:
            pct_low_veg = (df_sel["FCVC"] == 1).mean()
            txt_veg = "Baixo consumo de fibras." if pct_low_veg > 0.3 else "Bom consumo."
            plot_clinical_bar(df_sel, "FCVC", dict_veg, "Consumo de Vegetais", txt_veg)
            
        with c4:
            pct_low_water = (df_sel["CH2O"] == 1).mean()
            txt_water = "Risco de desidratação." if pct_low_water > 0.4 else "Hidratação OK."
            plot_clinical_bar(df_sel, "CH2O", dict_water, "Ingestão de Água", txt_water)
            
        with c5:
            pct_high_cal = (df_sel["FAVC"] == "Sim").mean()
            txt_cal = "Dieta hipercalórica." if pct_high_cal > 0.6 else "Controle OK."
            plot_clinical_bar(df_sel, "FAVC", dict_cal, "Alimentos Hipercalóricos", txt_cal)

        st.divider()

        # --- 7. LAUDO ANALÍTICO INTELIGENTE (PROFISSIONAL) ---
        st.subheader("📝 Relatório de Inteligência Clínica")

        # Cálculos das Variáveis para o Texto
        val_cal = (df_sel["FAVC"] == "Sim").mean() * 100
        val_sed = (df_sel["FAF"] == 0).mean() * 100
        val_fam = (df_sel["family_history"] == "Sim").mean() * 100
        val_wat = (df_sel["CH2O"] == 1).mean() * 100
        val_veg = (df_sel["FCVC"] == 1).mean() * 100
        
        # 1. Análise Dietética
        if val_cal > 60:
            diet_text = f"Observa-se um padrão dietético preocupante, onde **{val_cal:.1f}%** dos indivíduos relatam consumo frequente de alimentos hipercalóricos."
        else:
            diet_text = f"O padrão dietético mostra-se moderado, com **{val_cal:.1f}%** dos indivíduos consumindo alimentos hipercalóricos, indicando melhor controle nutricional relativo."

        # 2. Análise de Atividade Física
        if val_sed > 40:
            mov_text = f"Este cenário é agravado significativamente pelo sedentarismo, que atinge **{val_sed:.1f}%** da coorte, criando um balanço energético positivo favorável ao ganho de peso."
        else:
            mov_text = f"No entanto, o nível de sedentarismo é contido (**{val_sed:.1f}%**), o que atua como fator atenuante no risco metabólico global."

        # 3. Análise Genética/Hereditária
        if val_fam > 70:
            gen_text = f"A etiologia deste perfil apresenta **forte componente genético**, visto que **{val_fam:.1f}%** possuem histórico familiar de obesidade. Isso sugere menor responsividade a intervenções puramente comportamentais sem suporte clínico."
        elif val_fam > 40:
            gen_text = f"O fator hereditário é relevante (**{val_fam:.1f}%** com histórico), sugerindo uma etiologia multifatorial (genética + ambiente)."
        else:
            gen_text = f"A baixa prevalência de histórico familiar (**{val_fam:.1f}%**) sugere que a condição é **predominantemente comportamental/ambiental**, oferecendo excelente prognóstico com mudança de hábitos."

        # 4. Agravantes
        agravantes = []
        if val_wat > 40: agravantes.append(f"baixa hidratação ({val_wat:.0f}% bebem <1L)")
        if val_veg > 30: agravantes.append(f"pobreza de fibras na dieta ({val_veg:.0f}% não consomem vegetais)")
        
        if agravantes:
            risk_text = f"Fatores agravantes identificados: **{', '.join(agravantes)}**."
        else:
            risk_text = "Não foram identificados agravantes secundários críticos (hidratação e consumo de fibras adequados)."

        # --- Renderização do Texto ---
        with st.container():
            st.markdown(f"""
            ### Síntese Epidemiológica: {selected}
            
            **1. Padrão Comportamental e Metabólico**
            {diet_text} {mov_text}
            
            **2. Componente Hereditário**
            {gen_text}
            
            **3. Marcadores de Risco Adicionais**
            {risk_text}
            
            ---
            **Recomendação Estratégica:**
            Recomenda-se priorizar a **{'intervenção em estilo de vida (atividade física)' if val_sed > 50 else 'reeducação alimentar'}** como linha de frente, dado o perfil apresentado.
            """)

        # Tabela Final
        with st.expander("Ver Dados Detalhados"):
            cols_map = {"Age": "Idade", "Gender": "Gênero", "BMI": "IMC", TARGET_COL: "Diagnóstico", "family_history": "Hist. Familiar"}
            st.dataframe(df_sel.rename(columns=cols_map).head(50), use_container_width=True)

# =========================================================
# TAB 3: SOBRE 
# =========================================================
with tabs[2]:
    st.header("Sobre a Ferramenta")
    st.markdown("Informações sobre o sistema, arquitetura técnica e diretrizes de uso.")
    st.write("")

    # --- 1. AVISO ÉTICO (CRÍTICO) ---
    st.error(
        """
        **🚨 Aviso Importante - Uso Ético e Legal**
        
        Esta ferramenta é um protótipo desenvolvido para fins acadêmicos e de demonstração tecnológica. 
        **Ela não substitui avaliação médica profissional.** Os resultados são estimativas baseadas em padrões estatísticos populacionais e **não devem** ser usados como diagnóstico definitivo ou para prescrição de tratamentos sem validação clínica.
        """,
        icon="⚠️"
    )
    
    st.write("")

    # --- 2. O QUE É (CONTEXTO) ---
    with st.container(border=True):
        st.markdown("#### 📘 O Projeto")
        st.markdown(
            """
            Este Sistema de Triagem de Obesidade foi desenvolvido como parte do **Tech Challenge (Fase 4)**. 
            O objetivo é demonstrar a aplicação de técnicas avançadas de **Machine Learning** na área da saúde preventiva.
            
            A ferramenta traduz modelos matemáticos complexos em uma interface amigável, permitindo que profissionais de saúde e pesquisadores identifiquem rapidamente perfis de risco com base em antropometria e hábitos de vida.
            """
        )

    # --- 3. FICHA TÉCNICA (OS "DETALHES" QUE VOCÊ PEDIU) ---
    with st.container(border=True):
        st.markdown("#### ⚙️ Ficha Técnica do Modelo")
        
        # Métricas em destaque
        c_tech1, c_tech2, c_tech3, c_tech4 = st.columns(4)
        c_tech1.metric("Algoritmo", "Random Forest")
        c_tech2.metric("Acurácia", "97.87%")
        c_tech3.metric("Features", "17 Atributos")
        c_tech4.metric("Classes", "7 Níveis")
        
        st.divider()
        
        st.markdown("**Arquitetura e Processamento:**")
        st.markdown(
            """
            * **Modelo:** O núcleo do sistema utiliza um classificador *Random Forest* (Floresta Aleatória), escolhido por sua robustez em lidar com dados não-lineares e alta precisão em classificação multiclasse.
            * **Engenharia de Atributos:** O sistema calcula automaticamente o IMC (Índice de Massa Corporal) e trata variáveis categóricas para otimizar a predição.
            * **Dataset:** Baseado no conjunto de dados *'Estimation of obesity levels based on eating habits and physical condition'* (UCI Machine Learning Repository), contendo dados de indivíduos de países como México, Peru e Colômbia.
            * **Stack Tecnológica:** Python, Scikit-learn, Pandas, Matplotlib e Streamlit.
            """
        )

    # --- 4. COMO USAR ---
    with st.container(border=True):
        st.markdown("#### 🟢 Guia de Utilização")
        
        st.markdown(
            """
            1.  **Coleta de Dados (Aba Predição):** Preencha o formulário com dados reais do paciente. A precisão depende da veracidade das informações (peso, altura e hábitos honestos).
            2.  **Triagem Automática:** O sistema processa os dados em tempo real e retorna a categoria de peso estimada.
            3.  **Análise de Risco:** Verifique os alertas de cor (Verde, Amarelo, Vermelho) e leia a orientação clínica sugerida.
            4.  **Exploração Populacional (Aba Painel):** Use o painel analítico para entender tendências macroscópicas e comparar o paciente individual com a média do seu grupo de risco.
            """
        )

    # --- 5. LIMITAÇÕES ---
    with st.container(border=True):
        st.markdown("#### 🛡️ Limitações Conhecidas")
        st.markdown(
            """
            * **Dados Sintéticos:** Parte do dataset original foi gerada sinteticamente (SMOTE) para balanceamento de classes, o que pode introduzir vieses em casos de borda.
            * **IMC como Proxy:** O modelo baseia-se fortemente na relação Peso/Altura. Indivíduos com muita massa muscular (atletas) podem ser classificados incorretamente como "Sobrepeso" devido ao alto IMC, embora sejam saudáveis.
            * **Generalização:** O modelo foi treinado com dados demográficos específicos (Latino-americanos jovens/adultos) e pode ter menor precisão em populações com características muito distintas (ex: idosos ou crianças muito jovens).
            """
        )

    st.write("")
    st.markdown("---")
    st.caption("© 2025 Tech Challenge Data Analytics | Desenvolvido com ❤️ e Python.")