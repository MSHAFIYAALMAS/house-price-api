import streamlit as st
import torch
import torch.nn as nn

# ---------------------------------------------------
# PyTorch Model
# ---------------------------------------------------
class HouseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
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
# Page Config + CSS
# ---------------------------------------------------
st.set_page_config(page_title="Premium House Price Predictor", layout="centered")

st.markdown("""
    <style>
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #4F8EF7, #63E2FF);
            border-radius: 12px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
        }
        .card {
            background: rgba(255, 255, 255, 0.90);
            padding: 28px;
            border-radius: 14px;
            backdrop-filter: blur(6px);
            box-shadow: 0px 4px 18px rgba(0,0,0,0.12);
            margin-bottom: 20px;
        }
        .result-box {
            background-color: #E8F8F5;
            padding: 18px;
            border-radius: 12px;
            border-left: 6px solid #1ABC9C;
        }
        .footer {
            text-align: center;
            color: #888;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown("""
    <div class="header">
        <h1>🏙️ Premium House Price Prediction</h1>
        <p>Enter home details to get an estimated price</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("📏 Total Area (Sqft)", min_value=500, max_value=8000, value=1200)
    bhk = st.number_input("🛏️ BHK", min_value=1, max_value=10, value=3)
    bathroom = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2)

with col2:
    age = st.slider("🏚 Property Age (Years)", 0, 30, 5)
    furnishing = st.selectbox("🛋️ Furnishing Status", ["Unfurnished", "Semi-furnished", "Fully furnished"])
    location = st.selectbox(
        "📍 Location",
        ["Mumbai", "Bangalore", "Chennai", "Hyderabad", "Delhi", "Pune", "Kolkata"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Encoding for model
# ---------------------------------------------------
furnish_map = {
    "Unfurnished": 0,
    "Semi-furnished": 1,
    "Fully furnished": 2
}

location_map = {
    "Mumbai": 1,
    "Bangalore": 2,
    "Chennai": 3,
    "Hyderabad": 4,
    "Delhi": 5,
    "Pune": 6,
    "Kolkata": 7
}

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("🔍 Predict Price"):
    x = torch.tensor([[
        sqft,
        bhk,
        bathroom,
        age,
        furnish_map[furnishing]  # Only 5 features → matches model input
    ]], dtype=torch.float32)

    price = model(x).item()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.subheader("💰 Estimated House Price")
    st.success(f"₹ {price:,.2f}")
    st.write(f"📍 **Location:** {location}")
    st.write(f"🛋️ **Furnishing:** {furnishing}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("⚠️ Note: This model is untrained. Predictions may not be accurate.")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("<div class='footer'>Developed with ❤️ using Streamlit + PyTorch</div>", unsafe_allow_html=True)
