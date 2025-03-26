import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore

# Load dataset
def load_data(file):
    data = pd.read_csv(file, parse_dates=["timestamp"], index_col="timestamp")
    return data

# Anomaly detection methods
def detect_anomalies_isolation_forest(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = IsolationForest(contamination=0.05, random_state=42)
    data["anomaly"] = model.fit_predict(data_scaled)
    return data

def detect_anomalies_one_class_svm(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = OneClassSVM(nu=0.05)
    data["anomaly"] = model.fit_predict(data_scaled)
    return data

def detect_anomalies_lof(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    data["anomaly"] = model.fit_predict(data_scaled)
    return data

def detect_anomalies_dbscan(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = DBSCAN(eps=0.5, min_samples=5)
    data["anomaly"] = model.fit_predict(data_scaled)
    return data

def detect_anomalies_kmeans(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = KMeans(n_clusters=2, random_state=42)
    data["anomaly"] = model.fit_predict(data_scaled)
    return data

def detect_anomalies_zscore(data):
    data["z_score"] = np.abs(zscore(data))
    data["anomaly"] = data["z_score"].apply(lambda x: -1 if x > 3 else 1)
    return data

def detect_anomalies_stl(data):
    stl = STL(data.iloc[:, 0], seasonal=13)
    result = stl.fit()
    residuals = result.resid
    threshold = np.std(residuals) * 3
    data["anomaly"] = residuals.apply(lambda x: -1 if abs(x) > threshold else 1)
    return data

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def detect_anomalies_autoencoder(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    autoencoder = build_autoencoder(data_scaled.shape[1])
    autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=16, verbose=0)
    reconstructions = autoencoder.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    data["anomaly"] = np.where(mse > threshold, -1, 1)
    return data

# Streamlit UI
st.title("Anomaly Detection in Time Series Data")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("Data Preview:", data.head())
    
    method = st.selectbox("Select Anomaly Detection Method", [
        "Isolation Forest", "One-Class SVM", "Local Outlier Factor", "DBSCAN", "K-Means", "Z-Score", "STL Decomposition", "Autoencoder"
    ])
    
    if st.button("Detect Anomalies"):
        if method == "Isolation Forest":
            result = detect_anomalies_isolation_forest(data)
        elif method == "One-Class SVM":
            result = detect_anomalies_one_class_svm(data)
        elif method == "Local Outlier Factor":
            result = detect_anomalies_lof(data)
        elif method == "DBSCAN":
            result = detect_anomalies_dbscan(data)
        elif method == "K-Means":
            result = detect_anomalies_kmeans(data)
        elif method == "Z-Score":
            result = detect_anomalies_zscore(data)
        elif method == "STL Decomposition":
            result = detect_anomalies_stl(data)
        elif method == "Autoencoder":
            result = detect_anomalies_autoencoder(data)
        
        st.write("Anomaly Detection Completed.")
        
        # Plot anomalies
        fig, ax = plt.subplots()
        ax.plot(result.index, result.iloc[:, 0], label="Data")
        anomalies = result[result["anomaly"] == -1]
        ax.scatter(anomalies.index, anomalies.iloc[:, 0], color='red', label="Anomalies")
        ax.legend()
        st.pyplot(fig)

st.write("Upload a dataset to begin anomaly detection.")
