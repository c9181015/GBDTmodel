import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

# =========================
# 语言选择
# =========================
lang = st.sidebar.selectbox("Language / 语言", ["English", "中文"])

text = {
    "English": {
        "title": "Prediction Tool for Nosocomial Bacterial Infections in ACLF",
        "binary_title": "Binary Features (Yes/No)",
        "numeric_title": "Numerical Features",
        "predict_button": "Predict",
        "infection_prob": "Probability of Infection",
        "risk_result": "Risk Assessment",
        "high": "High Risk",
        "low": "Low Risk",
        "threshold": "Threshold",
        "show_shap": "Show SHAP Force Plot",
        "disclaimer": "Disclaimer: This result is for reference only and should not be used for diagnosis or treatment decisions.",
        "feature_labels": {
            "Diabetes": "Diabetes",
            "Cerebral Failure": "Cerebral Failure",
            "Respiratory Failure": "Respiratory Failure",
            "HE": "Hepatic Encephalopathy",
            "WBC": "White Blood Cells (×10⁹/L)",
            "INR": "INR",
            "Cr": "Creatinine (µmol/L)",
            "K": "Potassium (mmol/L)",
            "Na": "Sodium (mmol/L)",
            "TBIL": "Total Bilirubin (µmol/L)",
            "CRP": "C-reactive Protein (mg/L)",
            "ALB": "Albumin (g/L)",
        }
    },
    "中文": {
        "title": "ACLF院内细菌感染风险预测工具",
        "binary_title": "二分类特征（是/否）",
        "numeric_title": "数值型特征",
        "predict_button": "预测",
        "infection_prob": "院内感染概率",
        "risk_result": "风险评估",
        "high": "高风险",
        "low": "低风险",
        "threshold": "阈值",
        "show_shap": "显示 SHAP 图",
        "disclaimer": "免责声明：本结果仅供参考，不可作为诊断或治疗决策依据。",
        "feature_labels": {
            "Diabetes": "糖尿病",
            "Cerebral Failure": "脑衰竭",
            "Respiratory Failure": "呼吸衰竭",
            "HE": "肝性脑病",
            "WBC": "白细胞 (×10⁹/L)",
            "INR": "INR",
            "Cr": "肌酐 (µmol/L)",
            "K": "钾 (mmol/L)",
            "Na": "钠 (mmol/L)",
            "TBIL": "总胆红素 (µmol/L)",
            "CRP": "C反应蛋白 (mg/L)",
            "ALB": "白蛋白 (g/L)",
        }
    }
}

t = text[lang]

# =========================
# 加载模型
# =========================
MODEL_PATH = "GBDTmodel.pkl"
model = joblib.load(MODEL_PATH)

# 变量名保持不变
feature_names = [
    'Diabetes',
    'Cerebral Failure',
    'Respiratory Failure',
    'HE',
    'WBC',
    'INR',
    'Cr',
    'K',
    'Na',
    'TBIL',
    'CRP',
    'ALB'
]

# =========================
# 页面标题
# =========================
st.markdown(f"<h1 style='text-align: center;'>{t['title']}</h1>", unsafe_allow_html=True)

# =========================
# 输入界面
# =========================
user_input = {}

binary_features = ['Diabetes', 'Cerebral Failure', 'Respiratory Failure', 'HE']
st.subheader(t["binary_title"])

for feature in binary_features:
    label = t["feature_labels"][feature]
    choice = st.selectbox(label, ["No", "Yes"] if lang == "English" else ["否", "是"])
    user_input[feature] = 1 if choice in ["Yes", "是"] else 0

numeric_features = ['WBC', 'INR', 'Cr', 'K', 'Na', 'TBIL', 'CRP', 'ALB']
default_values = {
    'WBC': 6.0,
    'INR': 1.2,
    'Cr': 70.0,
    'K': 4.0,
    'Na': 138.0,
    'TBIL': 300.0,
    'CRP': 20.0,
    'ALB': 30.0
}

st.subheader(t["numeric_title"])

for feature in numeric_features:
    label = t["feature_labels"][feature]
    val = st.number_input(label, value=float(default_values[feature]))
    user_input[feature] = val

# =========================
# =========================
# 初始化 session_state
# =========================
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "input_df" not in st.session_state:
    st.session_state.input_df = None
if "class1_prob" not in st.session_state:
    st.session_state.class1_prob = None

# =========================
# 预测
# =========================
if st.button(t["predict_button"]):
    input_df = pd.DataFrame([[user_input[f] for f in feature_names]], columns=feature_names)

    predicted_proba = model.predict_proba(input_df)[0]
    class1_prob = predicted_proba[1] * 100

    st.session_state.predicted = True
    st.session_state.input_df = input_df
    st.session_state.class1_prob = class1_prob

# =========================
# 显示预测结果
# =========================
if st.session_state.predicted:
    input_df = st.session_state.input_df
    class1_prob = st.session_state.class1_prob

    st.write(f"**{t['infection_prob']}：** {class1_prob:.1f}%")

    threshold = 0.394
    risk = t["high"] if class1_prob / 100 >= threshold else t["low"]
    st.write(f"**{t['risk_result']}（{t['threshold']} {threshold:.3f}）：** {risk}")

    st.info(t["disclaimer"])

    # =========================
    # SHAP 可解释性
    # =========================
    if st.button(t["show_shap"]):
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(input_df)

        # 取当前样本
        single_exp = shap_explanation[0]

        # 手动固定 base value
        manual_base_value = -0.647

        # 用手动 base value 替换 explanation 中的 base_values
        single_exp.base_values = manual_base_value

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(single_exp, max_display=12, show=False)

        ax = plt.gca()

        # 调整字体
        ax.set_title("SHAP Waterfall Plot", fontsize=14)
        for label in ax.get_yticklabels():
            label.set_fontsize(12)
        for label in ax.get_xticklabels():
            label.set_fontsize(11)

        # 手动添加 base value 说明
        ax.text(
            0.02, 1.02,
            f"Base value = {manual_base_value:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            ha='left',
            va='bottom',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.25')
        )

        plt.tight_layout()
        st.pyplot(plt.gcf())

        if lang == "中文":
            with st.expander("🧩 点击查看 SHAP 瀑布图详细解释"):
                st.markdown("""
**SHAP 瀑布图（SHAP Waterfall Plot）** 用于解释单个样本的预测结果，展示各特征如何从基线值逐步推动模型输出变化。

**1️⃣ 基线值（Base Value）**  
- 这里固定显示为 `-0.647`。  
- 它表示模型在总体样本中的基准输出水平。  

**2️⃣ 最终预测值（f(x)）**  
- 瀑布图最右侧显示的是该样本的最终模型输出。  
- 它等于基线值加上所有特征的 SHAP 值贡献。  

**3️⃣ 特征贡献方向**  
- 正向贡献：将预测值往更高方向推动。  
- 负向贡献：将预测值往更低方向推动。  

**4️⃣ 特征贡献大小**  
- 条形越长，表示该特征对当前样本预测结果影响越大。  

**📘 总结**  
- 瀑布图比力图更稳定，也更适合精确展示单一样本的特征贡献。  
""")
        else:
            with st.expander("🧩 Click to view detailed SHAP Waterfall Plot explanation"):
                st.markdown("""
**SHAP Waterfall Plot** explains the prediction for a single sample by showing how each feature moves the model output step by step from the base value.

**1️⃣ Base Value**  
- The base value is fixed here as `-0.647`.  
- It represents the baseline output level of the model.  

**2️⃣ Final Output (f(x))**  
- The right side of the waterfall plot shows the final model output for the sample.  
- It equals the base value plus all SHAP contributions.  

**3️⃣ Direction of Contribution**  
- Positive contributions push the prediction higher.  
- Negative contributions push the prediction lower.  

**4️⃣ Magnitude of Contribution**  
- Longer bars indicate a stronger influence of that feature on the prediction.  

**📘 Summary**  
- The waterfall plot is more stable than the force plot and is better suited for precise display of feature contributions for an individual sample.  
""")
