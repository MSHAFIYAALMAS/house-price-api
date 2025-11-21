import streamlit as st
import torch
import torch.nn as nn

# ---------------------------------------------------
# PyTorch Model Definition
# ---------------------------------------------------
class HouseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = HouseModel()
    model.load_state_dict(torch.load("house_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------------------------------------------
# Custom Page Styling
# ---------------------------------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 38px;
            font-weight: 700;
            color: #4CAF50;
        }
        .sub {
            text-align: center;
            font-size: 18px;
            color: #666;
        }
        .input-box {
            background-color: #f7f7f7;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .result-box {
            background-color: #e7ffe7;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Title Section
# ---------------------------------------------------
st.markdown("<h1 class='main-title'>🏡 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Enter property details to estimate the price</p>", unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------
st.markdown("<div class='input-box'>", unsafe_allow_html=True)

sqft = st.number_input("📏 Square Feet", min_value=500, max_value=5000, value=1000)
bhk = st.number_input("🛏 BHK", min_value=1, max_value=10, value=2)

predict_btn = st.button("🔍 Predict Price")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Prediction Section
# ---------------------------------------------------
if predict_btn:
    x = torch.tensor([[sqft, bhk]], dtype=torch.float32)
    prediction = model(x).item()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    st.subheader("💰 Predicted Price")
    st.success(f"₹ {prediction:,.2f}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("⚠️ *Note: This is an untrained model. Predictions may not be accurate.*")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.write("---")
st.markdown("<p style='text-align:center;color:#999;'>Developed with ❤️ using PyTorch + Streamlit</p>", unsafe_allow_html=True)
