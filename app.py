# app.py — Streamlit dashboard for Student Productivity Classifier
#
# prerequisites: run train_model.py first
#   python train_model.py
#   streamlit run app.py

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Productivity Classifier",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 16px;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Prediction badge */
    .pred-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 50px;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin: 8px 0;
    }
    .badge-high   { background: linear-gradient(135deg,#16a34a,#15803d); color: white; }
    .badge-medium { background: linear-gradient(135deg,#d97706,#b45309); color: white; }
    .badge-low    { background: linear-gradient(135deg,#dc2626,#b91c1c); color: white; }

    /* Cards */
    .info-card {
        background: #1e2130;
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161825 !important; }

    /* Divider */
    hr { border-color: #2e3450; }

    /* Streamlit default tab */
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 600; }

    /* Remove extra padding */
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── load artifacts ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    with open("preprocessor.pkl", "rb") as f: preprocessor = pickle.load(f)
    with open("model.pkl",        "rb") as f: model        = pickle.load(f)
    with open("metadata.pkl",     "rb") as f: metadata     = pickle.load(f)
    return preprocessor, model, metadata

preprocessor, model, metadata = load_artifacts()

class_names       = metadata["class_names"]          # ["Low","Medium","High"]
ordinal_cols      = metadata["ordinal_cols"]
onehot_cols       = metadata["onehot_cols"]
feature_names_out = metadata["feature_names_out"]
model_comparison  = metadata["model_comparison"]
rf_importances    = metadata["rf_importances"]
branch_options    = metadata["branch_options"]
weekend_options   = metadata["weekend_options"]
study_labels      = metadata["weekly_study_labels"]
confusion_mat     = metadata["confusion_matrix"]

# emoji & colour maps
emoji_map  = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}
badge_map  = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}
color_map  = {"Low": "#ef4444", "Medium": "#f59e0b", "High": "#22c55e"}

# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📚 Student Productivity")
    st.markdown("#### ML Classifier Dashboard")
    st.markdown("---")

    st.markdown("**🏆 Best Model**")
    st.info(f"{metadata['best_model_name']}")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metadata['best_accuracy']:.1%}")
    c2.metric("Macro F1", f"{metadata['best_f1_macro']:.2f}")

    st.markdown("---")
    st.markdown("**📊 Dataset**")
    st.markdown("""
- **200** student responses  
- **13** input features  
- **3** productivity classes  
- **SMOTE** applied for class balance  
""")
    st.markdown("---")
    st.markdown("**🧪 Models Compared**")
    for name, scores in model_comparison.items():
        is_best = name == metadata["best_model_name"]
        star = "⭐ " if is_best else "   "
        st.markdown(f"`{star}{name}` → F1: **{scores['f1_macro']:.3f}**")

    st.markdown("---")
    st.caption("Built with Streamlit · Scikit-learn · Plotly")

# ── main header ────────────────────────────────────────────────────────────────

st.markdown("# 📚 Student Productivity Classifier")
st.markdown("Predict a student's **Overall Productivity** (Low / Medium / High) based on their study habits, attendance, and lifestyle inputs.")
st.markdown("---")

# ── tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Predict Productivity", "📊 Model Insights", "📈 Dataset Overview"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### 🎓 Enter Student Profile")
    st.markdown("Fill in the student details below and click **Predict** to get the productivity class.")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### 🏫 Academic Details")

        branch = st.selectbox(
            "Branch / Department",
            options=[b.title() for b in branch_options],
            index=0
        )
        year_of_study = st.selectbox(
            "Year of Study",
            options=["1st year", "2nd year"],
            index=0
        )
        class_hrs = st.selectbox(
            "Class Hours per Day",
            options=["4-5 hrs", "5-6 hrs", "6-7 hrs", "7+ hrs"],
            index=1
        )
        weekly_attendance = st.selectbox(
            "Weekly Attendance",
            options=["less than 50%", "about half", "most classes", "nearly all", "all classes"],
            index=2
        )
        academic_completion = st.selectbox(
            "Academic Work Completion",
            options=["none", "very little", "moderate", "high", "very high"],
            index=2
        )
        class_participation = st.selectbox(
            "Class Participation",
            options=["never", "rarely", "sometimes", "often", "always"],
            index=2
        )

    with col_r:
        st.markdown("#### 🌙 Lifestyle & Study Habits")

        sleep_hrs = st.selectbox(
            "Sleep Hours per Night",
            options=["<4 hrs", "4-5 hrs", "5-6 hrs", "6-7 hrs", "7-8 hrs", "8+ hrs"],
            index=3
        )
        screen_time = st.selectbox(
            "Screen Time per Day",
            options=["<4 hrs", "4-5 hrs", "5-6 hrs", "6+ hrs"],
            index=1
        )
        weekly_study_raw = st.selectbox(
            "Weekly Self-Study Hours",
            options=[1, 4, 7, 10],
            index=1,
            format_func=lambda x: study_labels[str(x)]
        )
        understanding = st.selectbox(
            "Understanding of Lectures",
            options=["very poor", "poor", "average", "good", "excellent"],
            index=2
        )
        study_mgmt = st.selectbox(
            "Study Time Management",
            options=["very poor", "poor", "average", "good", "excellent"],
            index=2
        )
        weekend_activity = st.selectbox(
            "Weekend Activity",
            options=weekend_options,
            index=0,
            format_func=lambda x: x.title()
        )

    st.markdown("")
    predict_btn = st.button("🔮 Predict Productivity", use_container_width=True, type="primary")

    if predict_btn:
        # build input dataframe in the exact column order expected by the preprocessor
        input_dict = {
            "Branch":                    branch.lower(),
            "Year of study":             year_of_study,
            "Class (hrs/day)":           class_hrs,
            "Sleep (hrs/night)":         sleep_hrs,
            "Weekend":                   weekend_activity,
            "Weekly Study (hrs)":        weekly_study_raw,
            "Academic work completion":  academic_completion,
            "Weekly Attendance":         weekly_attendance,
            "Understanding of lectures": understanding,
            "Study time management":     study_mgmt,
            "Class participation":       class_participation,
            "Screen Time (hrs/day)":     screen_time,
        }
        input_df = pd.DataFrame([input_dict])

        # preprocess
        input_proc = preprocessor.transform(input_df)

        # predict
        pred_index  = model.predict(input_proc)[0]
        pred_label  = class_names[pred_index]

        # probabilities (works for LR, SVM with probability=True, RF, etc.)
        try:
            probs = model.predict_proba(input_proc)[0]
        except AttributeError:
            probs = np.array([1/3, 1/3, 1/3])  # fallback

        confidence = probs[pred_index]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])

        with res_col1:
            st.markdown(
                f'<div class="info-card" style="text-align:center;">'
                f'<div style="font-size:0.85rem;color:#9ca3af;margin-bottom:4px;">PREDICTED PRODUCTIVITY</div>'
                f'<span class="pred-badge {badge_map[pred_label]}">'
                f'{emoji_map[pred_label]}  {pred_label.upper()}'
                f'</span>'
                f'<div style="font-size:0.85rem;color:#9ca3af;margin-top:8px;">Confidence: <b style="color:white">{confidence:.1%}</b></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with res_col2:
            st.metric("Model Used", metadata["best_model_name"])
            st.metric("Model Accuracy", f"{metadata['best_accuracy']:.1%}")

        with res_col3:
            st.metric("Predicted Class Index", pred_index)
            st.metric("Macro F1 Score", f"{metadata['best_f1_macro']:.2f}")

        # probability donut chart
        st.markdown("#### 📊 Probability Across All Classes")
        fig_prob = go.Figure(go.Bar(
            x=class_names,
            y=[float(p) for p in probs],
            marker_color=[color_map[c] for c in class_names],
            text=[f"{p:.1%}" for p in probs],
            textposition="outside"
        ))
        fig_prob.update_layout(
            paper_bgcolor="#1e2130",
            plot_bgcolor="#1e2130",
            font_color="#e5e7eb",
            height=300,
            yaxis=dict(range=[0, 1], gridcolor="#2e3450"),
            xaxis=dict(gridcolor="#2e3450"),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # show processed input
        with st.expander("🔬 See processed input (what the model actually received)"):
            proc_df = pd.DataFrame([input_proc[0]], columns=feature_names_out)
            st.dataframe(proc_df.T.rename(columns={0: "Encoded Value"}), use_container_width=True)

    else:
        st.info("👆 Fill in the student details above and click **Predict Productivity** to see the result.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    ins_col1, ins_col2 = st.columns(2, gap="large")

    with ins_col1:
        st.markdown("### 🏅 Model Comparison (Macro F1)")
        names_list = list(model_comparison.keys())
        f1_list    = [v["f1_macro"] for v in model_comparison.values()]
        acc_list   = [v["accuracy"] for v in model_comparison.values()]
        best_f1    = max(f1_list)

        colors_bar = [color_map["High"] if f == best_f1 else "#4f46e5" for f in f1_list]

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name="Macro F1", x=names_list, y=f1_list,
            marker_color=colors_bar,
            text=[f"{v:.3f}" for v in f1_list],
            textposition="outside"
        ))
        fig_compare.add_trace(go.Scatter(
            name="Accuracy", x=names_list, y=acc_list,
            mode="lines+markers",
            marker=dict(color="#f59e0b", size=8),
            line=dict(color="#f59e0b", width=2)
        ))
        fig_compare.add_hline(y=0.5, line_dash="dash", line_color="#ef4444",
                              annotation_text="0.5 baseline")
        fig_compare.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#e5e7eb", height=380,
            legend=dict(bgcolor="#1e2130"),
            yaxis=dict(range=[0, 1], gridcolor="#2e3450"),
            xaxis=dict(gridcolor="#2e3450"),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    with ins_col2:
        st.markdown("### 🗺️ Confusion Matrix")
        cm_array = np.array(confusion_mat)
        fig_cm = go.Figure(go.Heatmap(
            z=cm_array,
            x=class_names, y=class_names,
            colorscale="Blues",
            text=cm_array,
            texttemplate="%{text}",
            showscale=True
        ))
        fig_cm.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#e5e7eb",
            xaxis_title="Predicted", yaxis_title="Actual",
            height=380,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔑 Feature Importance (Random Forest)")

    imp_df = pd.DataFrame(
        list(rf_importances.items()),
        columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=True).tail(15)

    fig_imp = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation="h",
        marker=dict(
            color=imp_df["Importance"],
            colorscale="Viridis",
            showscale=True
        ),
        text=[f"{v:.3f}" for v in imp_df["Importance"]],
        textposition="outside"
    ))
    fig_imp.update_layout(
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
        font_color="#e5e7eb", height=480,
        xaxis=dict(gridcolor="#2e3450"),
        margin=dict(t=20, b=20, l=10, r=80)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Model comparison table
    st.markdown("### 📋 Full Model Comparison Table")
    compare_df = pd.DataFrame([
        {"Model": name, "Accuracy": f"{v['accuracy']:.4f}", "Macro F1": f"{v['f1_macro']:.4f}",
         "Best": "⭐" if name == metadata["best_model_name"] else ""}
        for name, v in model_comparison.items()
    ])
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🗄️ Dataset Overview — 200 Student Responses")

    ov_c1, ov_c2 = st.columns(2, gap="large")

    with ov_c1:
        st.markdown("#### 🎯 Productivity Class Distribution")
        class_dist = metadata["class_distribution"]
        order = ["Low", "Medium", "High"]
        labels = [k for k in order if k in class_dist]
        values = [class_dist[k] for k in labels]

        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker_colors=[color_map[l] for l in labels],
            hole=0.45,
            textinfo="label+percent+value"
        ))
        fig_pie.update_layout(
            paper_bgcolor="#1e2130", font_color="#e5e7eb",
            height=340, margin=dict(t=20, b=20),
            legend=dict(bgcolor="#1e2130")
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ov_c2:
        st.markdown("#### 🏫 Branch Distribution")
        branch_dist = metadata["branch_distribution"]
        b_labels = list(branch_dist.keys())
        b_values = list(branch_dist.values())

        fig_branch = go.Figure(go.Bar(
            x=b_values, y=b_labels, orientation="h",
            marker_color="#4f46e5",
            text=b_values, textposition="outside"
        ))
        fig_branch.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#e5e7eb", height=340,
            xaxis=dict(gridcolor="#2e3450"),
            margin=dict(t=20, b=20, l=10)
        )
        st.plotly_chart(fig_branch,use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📌 Feature Summary")
    feature_info = [
        {"Feature": "Branch",                   "Type": "Nominal",  "Unique Values": "8 branches"},
        {"Feature": "Year of Study",             "Type": "Ordinal",  "Unique Values": "1st / 2nd Year"},
        {"Feature": "Class (hrs/day)",           "Type": "Ordinal",  "Unique Values": "4-5, 5-6, 6-7, 7+"},
        {"Feature": "Sleep (hrs/night)",         "Type": "Ordinal",  "Unique Values": "<4 to 8+ hrs"},
        {"Feature": "Weekend Activity",          "Type": "Nominal",  "Unique Values": "5 activities"},
        {"Feature": "Weekly Study (hrs)",        "Type": "Numeric",  "Unique Values": "1, 4, 7, 10"},
        {"Feature": "Academic Work Completion",  "Type": "Ordinal",  "Unique Values": "none → very high"},
        {"Feature": "Weekly Attendance",         "Type": "Ordinal",  "Unique Values": "<50% → all classes"},
        {"Feature": "Understanding of Lectures", "Type": "Ordinal",  "Unique Values": "very poor → excellent"},
        {"Feature": "Study Time Management",     "Type": "Ordinal",  "Unique Values": "very poor → excellent"},
        {"Feature": "Class Participation",       "Type": "Ordinal",  "Unique Values": "never → always"},
        {"Feature": "Screen Time (hrs/day)",     "Type": "Ordinal",  "Unique Values": "<4 to 6+ hrs"},
        {"Feature": "Overall Productivity",      "Type": "Target 🎯", "Unique Values": "Low / Medium / High"},
    ]
    st.dataframe(pd.DataFrame(feature_info), use_container_width=True, hide_index=True)
