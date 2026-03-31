import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="SmartStroke AI", layout="wide")

# =========================
# FEATURES (MATCH TRAINING)
# =========================
core_features = [
    'age',
    'gender',
    'chest_pain',
    'high_blood_pressure',
    'irregular_heartbeat',
    'shortness_of_breath',
    'fatigue_weakness',
    'dizziness',
    'swelling_edema',
    'neck_jaw_pain',
    'excessive_sweating',
    'persistent_cough',
    'nausea_vomiting',
    'chest_discomfort',
    'cold_hands_feet',
    'snoring_sleep_apnea',
    'anxiety_doom'
]

ACTIONS = [
    ('high_blood_pressure', -1),
    ('fatigue_weakness', -1),
    ('dizziness', -1),
    ('snoring_sleep_apnea', -1),
    ('anxiety_doom', -1)
]

device = torch.device("cpu")

# =========================
# MODEL DEFINITIONS
# =========================
class StrokeGAT(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gat1 = GATConv(1, 64, heads=4, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(64 * 4)
        self.gat2 = GATConv(64 * 4, 32, heads=2, dropout=0.2)
        self.bn2 = nn.BatchNorm1d(32 * 2)

        self.classifier = nn.Sequential(
            nn.Linear(32 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = torch.nn.functional.elu(x)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = torch.nn.functional.elu(x)

        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1), x


class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return torch.softmax(self.net(state), dim=-1)


# =========================
# LOAD MODELS
# =========================
model = StrokeGAT(len(core_features))
model.load_state_dict(torch.load("gat_model.pt", map_location=device))
model.eval()

policy = PPOPolicy(len(core_features), len(ACTIONS))
policy.load_state_dict(torch.load("ppo_policy.pt", map_location=device))
policy.eval()

scaler = joblib.load("scaler.pkl")
edge_index = torch.load("edge_index.pt")

# =========================
# PREDICTION FUNCTION
# =========================
def model_predict(X_df):
    X_scaled = scaler.transform(X_df)

    data_list = []
    for row in X_scaled:
        x = torch.FloatTensor(row).unsqueeze(-1)
        data = Data(
            x=x,
            edge_index=edge_index,
            batch=torch.zeros(x.size(0), dtype=torch.long)
        )
        data_list.append(data)

    probs = []
    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            logit, _ = model(data)
            prob = torch.sigmoid(logit).item()
            probs.append(prob)

    return np.array(probs)

# =========================
# UI START
# =========================
st.title("🧠 SmartStroke AI")
st.caption("Predict • Explain • Recommend")

# =========================
# INPUT
# =========================
st.sidebar.header("Patient Input")

input_data = {}

# REAL inputs
input_data['age'] = st.sidebar.slider("Age", 0, 100, 30)

input_data['gender'] = st.sidebar.selectbox("Gender", ["Male", "Female"])
input_data['gender'] = 1 if input_data['gender'] == "Male" else 0

# binary symptoms
bin_features = [
    'chest_pain',
    'high_blood_pressure',
    'irregular_heartbeat',
    'shortness_of_breath',
    'fatigue_weakness',
    'dizziness',
    'swelling_edema',
    'neck_jaw_pain',
    'excessive_sweating',
    'persistent_cough',
    'nausea_vomiting',
    'chest_discomfort',
    'cold_hands_feet',
    'snoring_sleep_apnea',
    'anxiety_doom'
]

for feature in bin_features:
    input_data[feature] = st.sidebar.selectbox(
        feature.replace("_", " "),
        ["No", "Yes"]
    )
    input_data[feature] = 1 if input_data[feature] == "Yes" else 0

input_df = pd.DataFrame([input_data])

# Ensure correct column order
input_df = input_df[core_features]

# =========================
# ANALYSIS
# =========================
if st.sidebar.button("Analyze Patient"):

    col1, col2 = st.columns(2)

    # 🔮 Prediction
    with col1:
        st.subheader("🔮 Risk Prediction")

        prob = model_predict(input_df)[0]

        # % format (better)
        st.metric("Stroke Risk", f"{prob:.2%}")

        # Visual risk bar
        st.progress(float(prob))

        # Color-coded risk
        if prob < 0.3:
            st.success("🟢 Low Risk")
        elif prob < 0.7:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")

    # 📊 Explainability

    with col2:
        st.subheader("📊 Key Risk Factors")
    
        # Prepare tensor
        model.eval()

        patient_tensor = torch.FloatTensor(
            input_df.values[0]
        ).unsqueeze(-1).to(device)
        
        patient_data = Data(
            x=patient_tensor,
            edge_index=edge_index.to(device),
            batch=torch.zeros(patient_tensor.size(0), dtype=torch.long).to(device)
        )
        
        patient_data.x.requires_grad_(True)
        
        model.zero_grad()
        logit, _ = model(patient_data)
        logit.backward()
        
        importance = patient_data.x.grad.abs().squeeze().detach().cpu().numpy()
        importance = importance / (importance.sum() + 1e-8)
        
        top_idx = np.argsort(importance)[-5:]
        top_idx = top_idx[np.argsort(importance[top_idx])]
        
        features = [core_features[i].replace("_", " ") for i in top_idx]
        values = importance[top_idx]
        
            
        # 🔥 BAR CHART (NEW)
        import matplotlib.pyplot as plt
    
        fig, ax = plt.subplots()
        ax.barh(features, values)
        ax.set_xlabel("Importance")
        ax.set_title("Top Contributing Factors")
    
        st.pyplot(fig)
        plt.close(fig)
        # 🔥 TEXT SUMMARY (NEW)
        st.markdown("### 🔍 Key Insights")
    
        for i in reversed(top_idx):
            st.write(
                f"👉 **{core_features[i].replace('_',' ')}** "
                f"({importance[i]:.2%})"
            )

    # 💡 PPO Recommendations
    st.subheader("💡 Personalized Intervention Plan")
    
    state = input_df.values[0].copy()# ✅ FIXED
    
    initial_risk = prob
    
    steps = []
    risks = [initial_risk]
    feature_to_idx = {f: i for i, f in enumerate(core_features)}
    used_actions = set()
    policy.eval()
    with torch.no_grad():
        step = 0
        while len(steps) < 5 and step < 10:
            state_t = torch.FloatTensor(state).to(device)
        
            probs = policy(state_t)
        
            # Get top 3 actions
            topk = torch.topk(probs, k=3)
            
            action = None
            for a in topk.indices:
                if a.item() not in used_actions:
                    action = a.item()
                    break
            
            # fallback if all used
            if action is None:
                action = torch.argmax(probs).item()
            if action in used_actions:
                step+=1
                continue
        
            used_actions.add(action)
            feature, change = ACTIONS[action]
            idx = feature_to_idx[feature]
            decay = 1.0 / (len(steps) + 1)
            
            state[idx] = np.clip(state[idx] + change * decay,-3, 3)
            new_df = pd.DataFrame([state], columns=core_features)
            new_df = new_df[core_features]

            new_risk = model_predict(new_df)[0]
        
            steps.append((feature, new_risk))
            risks.append(new_risk)
            step+=1
        # 🔥 SHOW STEPS CLEANLY
        st.markdown("### 🧭 Recommended Actions")
        
        for i, (feature, r) in enumerate(steps):
            st.write(
                f"**Step {i+1}:** Improve **{feature.replace('_',' ')}** → Risk: {r:.2%}"
            )
        
        # 🔥 RISK TREND GRAPH
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot(range(len(risks)), risks, marker='o')
        ax.set_title("Risk Reduction Over Steps")
        ax.set_xlabel("Step")
        ax.set_ylabel("Risk")
        
        st.pyplot(fig)
        plt.close(fig)
        # 🔥 FINAL SUMMARY BOX
        final_risk = risks[-1] 
        reduction = initial_risk - final_risk
        
        st.success(
            f"Final Risk: {reduction:.2%}  |  Reduction: {final_risk:.2%}"
        )
        
        # 🔥 INTERPRETATION
        if reduction > 0.3:
            st.success("Excellent improvement with lifestyle changes 🚀")
        elif reduction > 0.15:
            st.warning("Moderate improvement possible ⚠️")
        else:
            st.error("Limited improvement — medical attention advised ❗")
