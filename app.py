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
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    model = HouseModel()
    model.load_state_dict(torch.load("house_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Smart Home Valuator", layout="wide")

# Custom CSS for New UI
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.header-box {
    background: linear-gradient(135deg, #1e88e5, #42a5f5);
    padding: 28px 20px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}
.price-box {
    background: #e3f2fd;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("<div class='header-box'><h1>🏡 Smart Home Price Valuator</h1><p>AI-powered real estate price predictor</p></div>", unsafe_allow_html=True)

# ---------------------------------------------------
# LAYOUT
# ---------------------------------------------------
left, right = st.columns([1.2, 1])

with left:
    st.markdown("### 📌 Enter Property Information")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sqft = st.slider("Total Area (sqft)", 400, 6000, 1200)
    bhk = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5])
    location = st.selectbox("Location", ["Bangalore", "Hyderabad", "Chennai", "Mumbai", "Pune"])

    age = st.number_input("Age of House (Years)", 0, 30, 5)
    furnished = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
    balcony = st.selectbox("Balconies", [0, 1, 2, 3])
    parking = st.selectbox("Parking", ["Yes", "No"])

    submit = st.button("🔍 Predict Price", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("### 💰 Estimated Value")
    st.markdown("<div class='price-box'>", unsafe_allow_html=True)

    if submit:
        x = torch.tensor([[sqft, bhk]], dtype=torch.float32)
        predicted_value = model(x).item()
        final_price = predicted_value * 5000

        st.success(f"## ₹ {final_price:,.2f}")
    else:
        st.info("Fill details and click *Predict Price*.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.write("---")
st.caption("🔧 Built with Streamlit • Powered by PyTorch • Designed for Real Estate Intelligence")
