import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time

# Load pre-trained model
model = load_model("gasModelTest.h5")

# Load the scaler used for normalization during training
scaler = StandardScaler()  # Assuming you've saved the scaler, you should load the scaler if it's saved

# Streamlit UI for input
st.title("Gas Leak Detection System")

# Create empty containers for displaying results
status_container = st.empty()
lpg_status_container = st.empty()
butana_status_container = st.empty()
metana_status_container = st.empty()

# Simulate the automatic collection of sensor data (replace this with actual data collection logic)
def get_sensor_data():
    # Simulate reading new data, for example from a file or an MQTT broker
    # Here, we generate random values as placeholders for sensor data
    lpg = np.random.randint(0, 11)
    butana = np.random.randint(0, 11)
    metana = np.random.randint(0, 11)
    return lpg, butana, metana

# Function to classify gas readings
def classify(value):
    if 0 <= value <= 5:
        return "Aman"
    elif 5 < value <= 6:
        return "Waspada"
    elif 6 < value <= 10:
        return "Kebocoran"
    return "Tidak valid"

# Use session_state to hold the previous status
if 'status' not in st.session_state:
    st.session_state.status = ""

# Main loop for updating predictions every few seconds
while True:
    # Get new sensor data
    lpg, butana, metana = get_sensor_data()

    # Classify each sensor value
    lpg_status = classify(lpg)
    butana_status = classify(butana)
    metana_status = classify(metana)

    # Determine the overall status
    if lpg_status == "Aman" and butana_status == "Aman" and metana_status == "Aman":
        status = "Aman"
    elif lpg_status == "Kebocoran" or butana_status == "Kebocoran" or metana_status == "Kebocoran":
        status = "Kebocoran"
    else:
        status = "Waspada"

    # If the status has changed, update the session state and display the new status
    if st.session_state.status != status:
        st.session_state.status = status

        # Display the result in the UI
        status_container.write(f"Predicted Status: {status}")
        lpg_status_container.write(f"LPG Status: {lpg_status}")
        butana_status_container.write(f"BUTANA Status: {butana_status}")
        metana_status_container.write(f"METANA Status: {metana_status}")

        # Prepare the input data for the model
        input_data = pd.DataFrame([[lpg, butana, metana]], columns=["LPG", "BUTANA", "METANA"])

        # Normalize the input data (use the same scaler as during training)
        input_scaled = scaler.fit_transform(input_data)  # If you saved the scaler, load it and use it instead

        # Make prediction using the trained model
        prediction = model.predict(input_scaled)

        # Format the data into '00.00.00' format
        sensor_values_str = f"{lpg:02d}.{butana:02d}.{metana:02d}"

        # Display model's predicted status
        status_container.write(f"Model Predicted Status: {sensor_values_str}")

    # Wait before refreshing (e.g., 5 seconds)
    time.sleep(1)

    # Optionally, if you want the UI to be updated, just loop here to simulate real-time updates
    # (not recommended to call st.experimental_rerun() directly)
