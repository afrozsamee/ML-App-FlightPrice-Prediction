import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

le = LabelEncoder()

st.title("✈️ Flight Price Prediction App")
st.info("Predict flight ticket prices using XGBoost!")

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_excel("https://github.com/afrozsamee/Predictions_price_of_FlightTickets/raw/master/Data_Train.xlsx")
    return df

df = load_data()

# ========== DATA PREVIEW ==========
with st.expander("Dataset Preview"):
    st.dataframe(df.head())

# ========== FEATURE ENGINEERING ==========
def preprocess(data):
    data = data.copy()
    data.dropna(inplace=True)

    # Date_of_Journey
    data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'])
    data['Journey_Day'] = data['Date_of_Journey'].dt.day
    data['Journey_Month'] = data['Date_of_Journey'].dt.month
    data.drop('Date_of_Journey', axis=1, inplace=True)

    # Dep_Time
    data['Dep_Time'] = pd.to_datetime(data['Dep_Time'])
    data['Dep_Hour'] = data['Dep_Time'].dt.hour
    data['Dep_Min'] = data['Dep_Time'].dt.minute
    data.drop('Dep_Time', axis=1, inplace=True)

    # Arrival_Time
    data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'])
    data['Arr_Hour'] = data['Arrival_Time'].dt.hour
    data['Arr_Min'] = data['Arrival_Time'].dt.minute
    data.drop('Arrival_Time', axis=1, inplace=True)

    # Duration
    data['Duration'] = data['Duration'].astype(str)
    data['Dur_Hour'] = data['Duration'].str.extract(r'(\d+)h').fillna(0).astype(int)
    data['Dur_Min'] = data['Duration'].str.extract(r'(\d+)m').fillna(0).astype(int)
    data.drop('Duration', axis=1, inplace=True)

    # Encode categorical columns
    cat_cols = ["Airline", "Source", "Destination", "Route", "Total_Stops", "Additional_Info"]
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])

    return data

df_processed = preprocess(df)

X = df_processed.drop("Price", axis=1)
y = df_processed["Price"]

# ========== SIDEBAR INPUTS ==========
with st.sidebar:
    st.header("✈️ Enter Flight Details")

    Airline = st.selectbox("Airline", df["Airline"].unique())
    Source = st.selectbox("Source", df["Source"].unique())
    Destination = st.selectbox("Destination", df["Destination"].unique())
    Total_Stops = st.selectbox("Stops", df["Total_Stops"].unique())
    Additional_Info = st.selectbox("Additional Info", df["Additional_Info"].unique())

    # Raw date input
    Date_of_Journey = st.date_input("Date of Journey")

    # Time selectors formatted as the dataset requires
    Dep_Time = st.time_input("Departure Time")
    Arrival_Time = st.time_input("Arrival Time")

    # Duration (raw format to match dataset)
    Dur_Hour = st.number_input("Duration Hours", min_value=0, max_value=50, value=2)
    Dur_Min = st.number_input("Duration Minutes", min_value=0, max_value=59, value=30)

    Duration = f"{int(Dur_Hour)}h {int(Dur_Min)}m"

    input_data = {
        "Airline": Airline,
        "Date_of_Journey": Date_of_Journey.strftime("%d/%m/%Y"),
        "Source": Source,
        "Destination": Destination,
        "Route": "Not Available",
        "Dep_Time": Dep_Time.strftime("%H:%M"),
        "Arrival_Time": Arrival_Time.strftime("%H:%M"),
        "Duration": Duration,
        "Total_Stops": Total_Stops,
        "Additional_Info": Additional_Info
    }

    input_df = pd.DataFrame([input_data])

# Preprocess input by appending to df
input_transformed = preprocess(pd.concat([df.drop("Price", axis=1), input_df], axis=0)).tail(1)


# ========== SHOW INPUT ==========
with st.expander("User Input Preview"):
    st.dataframe(input_df)

# ========== MODEL TRAINING ==========
model = XGBRegressor(n_estimators=500, learning_rate=0.1)
model.fit(X, y)

# ========== PREDICTION ==========
# align columns to training data
input_transformed = input_transformed[X.columns]
price_pred = model.predict(input_transformed)[0]

# ========== RESULT ==========
st.subheader("💰 Predicted Flight Price")
st.success(f"₹ {price_pred:,.2f}")
