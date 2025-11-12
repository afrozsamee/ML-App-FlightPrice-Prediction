import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# =====================
# 🎯 PAGE CONFIG
# =====================
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")

st.title("✈️ Flight Price Prediction App")
st.caption("Built using Streamlit + XGBoost")
st.info("Predict flight ticket prices and view model validation & explainability insights.")

# =====================
# 📂 LOAD DATA
# =====================

@st.cache_data
def load_data():
    df = pd.read_excel("https://github.com/afrozsamee/Predictions_price_of_FlightTickets/raw/master/Data_Train.xlsx")
    return df


df = load_data()

with st.expander("📊 Dataset Preview"):
    st.dataframe(df.head())

# =====================
# ⚙️ PREPROCESS FUNCTION
# =====================
le = LabelEncoder()

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

# =====================
# 🔢 TRAIN-TEST SPLIT
# =====================
X = df_processed.drop("Price", axis=1)
y = df_processed["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================
# 🧠 TRAIN MODEL
# =====================
model = XGBRegressor(n_estimators=500, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# =====================
# 📈 MODEL VALIDATION
# =====================
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# =====================
# 🧭 TABS LAYOUT
# =====================
tab1, tab2, tab3 = st.tabs(["💡 Predict", "📊 Validation", "🔍 Explainability"])

# =====================
# ✈️ TAB 1 — PREDICTION
# =====================
with tab1:
    st.header("💡 Enter Flight Details")

    with st.sidebar:
        st.subheader("🧾 Flight Information")

        Airline = st.selectbox("Airline", df["Airline"].unique())
        Source = st.selectbox("Source", df["Source"].unique())
        Destination = st.selectbox("Destination", df["Destination"].unique())
        Total_Stops = st.selectbox("Stops", df["Total_Stops"].unique())
        Additional_Info = st.selectbox("Additional Info", df["Additional_Info"].unique())

        Date_of_Journey = st.date_input("Date of Journey")
        Dep_Time = st.time_input("Departure Time")
        Arrival_Time = st.time_input("Arrival Time")

        Dur_Hour = st.number_input("Duration Hours", min_value=0, max_value=50, value=2)
        Dur_Min = st.number_input("Duration Minutes", min_value=0, max_value=59, value=30)
        Duration = f"{int(Dur_Hour)}h {int(Dur_Min)}m"

        # Input as DataFrame
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
            "Additional_Info": Additional_Info,
        }
        input_df = pd.DataFrame([input_data])

    

    if st.button("🎯 Predict Price"):
        input_transformed = preprocess(pd.concat([df.drop("Price", axis=1), input_df], axis=0)).tail(1)
        input_transformed = input_transformed[X.columns]
        price_pred = model.predict(input_transformed)[0]

        with st.expander("🧾 Input Summary"):
        st.dataframe(input_df)

        st.subheader("💰 Predicted Flight Price")
        st.success(f"Estimated Price: ₹ {price_pred:,.2f}")

        # Downloadable report
        import io
        output = io.StringIO()
        input_df["Predicted Price"] = [price_pred]
        input_df.to_csv(output, index=False)
        st.download_button(
            "⬇️ Download Prediction Report",
            data=output.getvalue(),
            file_name="flight_price_prediction.csv",
            mime="text/csv",
        )

# =====================
# 📊 TAB 2 — VALIDATION
# =====================
with tab2:
    st.header("📊 Model Performance Validation")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:,.0f}")
    col3.metric("RMSE", f"{rmse:,.0f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Flight Prices")
    st.pyplot(fig)

# =====================
# 🔍 TAB 3 — EXPLAINABILITY
# =====================
with tab3:
    st.header("🔍 Feature Importance (Explainability)")
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(
        "Importance", ascending=False
    )
    st.bar_chart(feat_imp.set_index("Feature"))
    st.caption("Features with higher importance have a stronger impact on price prediction.")
