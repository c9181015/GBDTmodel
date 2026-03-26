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
        import matplotlib.transforms as mtransforms
        from matplotlib.patches import Rectangle
        from matplotlib.ticker import FormatStrFormatter

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            shap_values_for_sample = shap_values[1][0]
            base_value = explainer.expected_value[1]
        else:
            shap_values_for_sample = shap_values[0]
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = np.atleast_1d(base_value)[-1]

        base_value = float(base_value)

        # 你想手动显示的值
        manual_base_value = -0.647

        plt.figure(figsize=(12, 10))
        shap.force_plot(
            base_value,
            shap_values_for_sample,
            input_df.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )

        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # 先尝试隐藏 SHAP 自动生成的近似值文字
        for txt in ax.texts:
            txt_str = txt.get_text().strip()
            if txt_str in [f"{base_value:.1f}", f"{base_value:.2f}", f"{base_value:.3f}"]:
                txt.set_visible(False)

        # 用“x按数据坐标，y按轴坐标”的混合坐标，确保文字正对灰色虚线
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

        # 在原 base value 上方盖一个白底小框，尽量遮住 SHAP 默认文字
        cover_width = 0.9   # 这个值可微调；如果没盖住就改大一点，如 1.1
        cover_box = Rectangle(
            (base_value - cover_width / 2, 0.935),
            cover_width,
            0.08,
            transform=trans,
            facecolor='white',
            edgecolor='white',
            zorder=1000,
            clip_on=False
        )
        ax.add_patch(cover_box)

        # 重新画竖线
        ax.axvline(base_value, color='gray', linestyle='--', linewidth=1.2, zorder=1001)

        # 手动把 base value 写在竖线正上方
        ax.text(
            base_value,
            0.975,
            f"{manual_base_value:.3f}",
            transform=trans,
            color='black',
            fontsize=13,
            ha='center',
            va='top',
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'),
            zorder=1002,
            clip_on=False
        )

        for label in ax.get_yticklabels():
            label.set_fontsize(14)
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
        for txt in ax.texts:
            txt.set_fontsize(11)

        plt.tight_layout()
        st.pyplot(plt.gcf())

        if lang == "中文":
            with st.expander("🧩 点击查看 SHAP 力图详细解释"):
                st.markdown("""
**SHAP 力图（SHAP Force Plot）** 用于解释单个样本的预测结果，展示每个特征对模型输出的影响。

**1️⃣ 基线值（Base Value）**  
- 图中手动标注的基线值为 `-0.647`。  

**2️⃣ 模型输出值（f(x)）**  
- 图中显示的 *f(x)* 值是该样本的最终预测结果。  
- 它等于基线值加上所有特征的 SHAP 值：  
  `f(x) = base value + Σ(SHAP_i)`  

**3️⃣ 特征贡献（红色和蓝色箭头）**  
- 🔴 **红色箭头**：对预测结果有正向贡献（推高预测值）。  
- 🔵 **蓝色箭头**：对预测结果有负向贡献（降低预测值）。  

**4️⃣ 影响程度（箭头长度）**  
- 箭头越长，说明该特征的 SHAP 值绝对值越大，对当前样本预测的影响越显著。  
""")
        else:
            with st.expander("🧩 Click to view detailed SHAP Force Plot explanation"):
                st.markdown("""
**SHAP Force Plot** explains the prediction for an individual sample by showing how each feature contributes to the model output.

**1️⃣ Base Value**  
- The manually displayed base value is `-0.647`.  

**2️⃣ Model Output (f(x))**  
- The *f(x)* is the final predicted value for this sample.  
- It equals the base value plus all SHAP values:  
  `f(x) = base value + Σ(SHAP_i)`  

**3️⃣ Feature Contributions**  
- Red arrows push the prediction higher.  
- Blue arrows push the prediction lower.  

**4️⃣ Magnitude of Impact**  
- Longer arrows indicate stronger feature influence on the prediction.
""")
