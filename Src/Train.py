import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

# Load the latest CSV data with optimization
def load_data():
    df = pd.read_csv("testing_server.csv", header=None)

    # Function to check if the data can be converted to integers
    def is_valid_data(data):
        try:
            parts = data.split(".")
            if len(parts) == 3:
                # Check if all parts can be converted to integers
                lpg, butana, metana = map(int, parts)
                return True
        except ValueError:
            return False
        return False

    # Filter only valid data
    valid_data = df[0].apply(is_valid_data)

    # Apply the transformation only to valid rows
    df_valid = df[valid_data]

    # Split the valid sensor values into 3 columns (LPG, BUTANA, METANA)
    df_split = df_valid[0].str.split(".", expand=True)
    df_split.columns = ["LPG", "BUTANA", "METANA"]
    df_split[["LPG", "BUTANA", "METANA"]] = df_split[["LPG", "BUTANA", "METANA"]].astype(int)

    # Classify the status based on the sensor readings
    def get_status(row):
        lpg_status = "Aman" if row['LPG'] <= 5 else "Waspada" if row['LPG'] <= 6 else "Kebocoran"
        butana_status = "Aman" if row['BUTANA'] <= 5 else "Waspada" if row['BUTANA'] <= 6 else "Kebocoran"
        metana_status = "Aman" if row['METANA'] <= 5 else "Waspada" if row['METANA'] <= 6 else "Kebocoran"

        if lpg_status == "Aman" and butana_status == "Aman" and metana_status == "Aman":
            return "Aman"
        elif lpg_status == "Kebocoran" or butana_status == "Kebocoran" or metana_status == "Kebocoran":
            return "Kebocoran"
        else:
            return "Waspada"

    df_split["status"] = df_split.apply(get_status, axis=1)
    return df_split

# Train the model
def train_model():
    df = load_data()

    # Encode the status labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["status"])

    # Features (LPG, BUTANA, METANA)
    X = df[["LPG", "BUTANA", "METANA"]].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build the model
    model = Sequential([
        Dense(16, activation='relu', input_shape=(3,)),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=4)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the model
    model.save("gasModelTest.h5")

    return model

# Check data size for periodic training
def check_data_size_and_train():
    while True:
        df = pd.read_csv("testing_server.csv", header=None)

        if len(df) >= 10:  # Wait until there are at least 10 new data points
            print("Training model...")
            train_model()  # Train model when new data is sufficient
        else:
            print(f"Not enough data to train: {len(df)} records found.")

        time.sleep(10)  # Check for new data every 10 seconds

# Start checking data and training
if __name__ == "__main__":
    check_data_size_and_train()
