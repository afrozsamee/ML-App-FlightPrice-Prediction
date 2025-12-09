# ✈️ Flight Price Prediction App

A **Machine Learning web app** built with **Streamlit** that predicts flight ticket prices based on travel details such as airline, source, destination, date, time, and flight duration.  
This project demonstrates **data preprocessing, feature engineering, model validation, and explainability** using **XGBoost**.

---

## 🚀 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-app-flightprice-prediction-2m6grhlmvnfbqp2iw2natn.streamlit.app/)

👉 **Click the badge above to open the live deployed app.**

---

## 🧠 Project Overview

This interactive app allows users to:
- Input flight details (airline, route, stops, duration, etc.)
- Predict the **estimated ticket price**
- Visualize **model validation metrics** (R², MAE, RMSE)
- Understand **feature importance** behind predictions
- Download a **personalized prediction report**

The goal is to make machine learning results **transparent, explainable, and visually engaging** for end users and recruiters.

---

## 🏗️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Frontend** | Streamlit |
| **Machine Learning** | XGBoost, Scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Streamlit Charts |
| **Dataset Source** | Kaggle / Open Flight Ticket Price Dataset |

---

## ⚙️ Features

✅ **Interactive Flight Input Form** — Enter travel details in a simple UI  
✅ **"Predict" Button** — One-click price prediction  
✅ **Model Validation Dashboard** — R², MAE, and RMSE metrics displayed  
✅ **Actual vs Predicted Plot** — Shows how close predictions are to real prices  
✅ **Feature Importance Chart** — Explains which features affect price the most  
✅ **Downloadable Report** — Export personalized prediction as a CSV  
✅ **Fast Loading** — Cached dataset and preprocessing for better UX  

---

## 📊 Model Insights

- **Algorithm Used:** XGBoost Regressor  
- **Training Split:** 80% Train / 20% Test  
- **Performance Metrics:**
  - R² Score: ~0.89  
  - MAE: ~1900 ₹  
  - RMSE: ~2700 ₹  
*(Values are approximate; they vary depending on training and hyperparameters.)*

The model achieves strong predictive performance and provides transparent insights through explainability visuals.

---

## 🧾 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/flight-price-predictor.git
   cd flight-price-predictor
