import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# Load the trained model
model = load_model("best_ipl_model")

# Load and normalize Excel data
players_df = pd.read_excel(r"C:\Users\User\OneDrive\Desktop\Player data\Dataset\2023.csv.xlsx")

# Normalize column names
players_df.columns = players_df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")

# Remove unwanted columns
players_df = players_df.loc[:, ~players_df.columns.str.contains("^unnamed", case=False)]
players_df = players_df.loc[:, ~players_df.columns.str.lower().isin(['nan', '', 'unnamed:_0'])]

# Rename necessary columns
rename_dict = {
    'player': 'Player',
    'runs': 'Runs',
    'mat': 'Mat',
    'inns': 'Inns',
    'no': 'NO',
    'avg': 'Avg',
    'bf': 'BF',
    'sr': 'SR',
    '100': '100',
    '50': '50',
    '4s': '4s',
    '6s': '6s',
    'bow_mat': 'bow_mat',
    'bow_inns': 'bow_inns',
    'ov': 'Ov',
    'bow_runs': 'bow_runs',
    'wkts': 'Wkts',
    'bow_avg': 'bow_avg',
    'econ': 'Econ',
    'bow_sr': 'bow_sr',
    '4w': '4w',
    '5w': '5w'
}

# Rename Player column if it exists
if 'player' in players_df.columns:
    rename_dict['player'] = 'Player'
elif 'player_name' in players_df.columns:
    rename_dict['player_name'] = 'Player'

players_df.rename(columns=rename_dict, inplace=True)

# 🚨 Stop if Player column still doesn't exist
if 'Player' not in players_df.columns:
    st.error("❌ 'Player' column not found in the data. Please check your Excel sheet.")
    st.stop()

# Strip whitespaces in Player names
players_df['Player'] = players_df['Player'].astype(str).str.strip()

# Drop auction_price column if it exists
if 'auction Price' in players_df.columns:
    players_df.drop(columns=['auction Price'], inplace=True)

# Required input features for model
required_features = list(model.feature_names_in_)  # safer than hardcoding

# Fill missing features with 0.0
for col in required_features:
    if col not in players_df.columns:
        players_df[col] = 0.0

# Helper function to run prediction and rename output column
def predict_and_rename(model, data):
    try:
        input_data = data[model.feature_names_in_]  # ensure correct columns
        prediction = predict_model(estimator=model, data=input_data)

        if "Label" in prediction.columns:
            prediction.rename(columns={"Label": "prediction_label"}, inplace=True)
        return prediction
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return pd.DataFrame()

# Streamlit UI
st.title("🏏 IPL Auction Price Prediction")

option = st.radio("Select Player Type", ["Choose Existing Player", "Add New Player"])

# EXISTING PLAYER MODE
if option == "Choose Existing Player":
    player_name = st.selectbox("Select a Player", players_df["Player"].unique())

    if player_name:
        player_data = players_df[players_df["Player"].str.strip() == player_name.strip()]

        if player_data.empty:
            st.error("❌ No stats found for this player. Please check the name formatting.")
            st.stop()

        st.subheader("📊 Player Stats")
        st.write(player_data)

        if st.button("Predict Auction Price"):
            prediction = predict_and_rename(model, player_data)
            if "prediction_label" in prediction.columns:
                price = prediction["prediction_label"].values[0]
                st.success(f"💸 prediction_label: ₹ {price:,.2f}")
            else:
                st.error("❌ Prediction failed: 'prediction_label' column missing in model output.")

# NEW PLAYER MODE
else:
    st.subheader("🆕 Enter New Player Stats")
    input_data = {}

    for feature in required_features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

    new_player_df = pd.DataFrame([input_data])

    if st.button("Predict for New Player"):
        prediction = predict_and_rename(model, new_player_df)
        if "prediction_label" in prediction.columns:
            price = prediction["prediction_label"].values[0]
            st.success(f"💸 prediction_label: ₹ {price:,.2f}")
        else:
            st.error("❌ Prediction failed: 'prediction_label' column missing in model output.")
