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
            nn.Linear(6, 32),
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
# Page Styling
# ---------------------------------------------------
st.set_page_config(page_title="Advanced House Price Predictor", layout="centered")

st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: 700;
            color: #2E86C1;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #555;
        }
        .card {
            background: #ffffff;
            padding: 25px;
            border-radius: 14px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.12);
            margin-bottom: 20px;
        }
        .result-box {
            background-color: #E8F8F5;
            padding: 18px;
            border-radius: 12px;
            border-left: 6px solid #1ABC9C;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.markdown("<h1 class='title'>🏙️ Advanced House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Provide complete property details to estimate the house price</p>", unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------
# Input Form
# ---------------------------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        sqft = st.number_input("📏 Total Area (Sqft)", min_value=500, max_value=8000, value=1200)
        bhk = st.number_input("🛏️ BHK", min_value=1, max_value=10, value=3)
        bathroom = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2)

    with col2:
        carpet = st.number_input("📐 Carpet Area (Sqft)", min_value=300, max_value=5000, value=900)
        age = st.slider("🏚 Property Age (Years)", 0, 30, 5)
        furnishing = st.selectbox("🛋️ Furnishing Status", ["Unfurnished", "Semi-furnished", "Fully furnished"])

    # Location Input
    location = st.selectbox(
        "📍 Location",
        ["Mumbai", "Bangalore", "Chennai", "Hyderabad", "Delhi", "Pune", "Kolkata"]
    )

    # Amenities
    amenities = st.multiselect(
        "✨ Extra Amenities",
        ["Parking", "Lift", "Security", "Power Backup", "Gym", "Swimming Pool"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Convert Inputs
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

amenity_score = len(amenities)  # simple scoring, can be improved

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------
predict = st.button("🔍 Predict House Price")

if predict:
    x = torch.tensor([[
        sqft,
        bhk,
        bathroom,
        carpet,
        age,
        amenity_score
    ]], dtype=torch.float32)

    price = model(x).item()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    st.subheader("💰 Estimated Price")
    st.success(f"₹ {price:,.2f}")

    st.write(f"**Location:** {location}  ")
    st.write(f"**Furnishing:** {furnishing}  ")
    st.write(f"**Amenities Included:** {', '.join(amenities) if amenities else 'None'}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("⚠️ This model is not trained — numbers are placeholders.")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.write("---")
st.markdown("<p style='text-align:center;color:#888;'>Developed with ❤️ using Streamlit + PyTorch</p>", unsafe_allow_html=True)
