# Supply Chain Management Demand Forecasting Project

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow import keras
from tensorflow.keras import layers

# Step 2: Load Dataset
print("Loading dataset...")
data = pd.read_csv("C:/Users/ch.Tharani/Desktop/Project 2/supply_chain_data.csv") # Make sure your CSV file is in the same directory
print("Initial data preview:")
print(data.head())

# Step 3: Data Cleaning
print("\nCleaning data...")
data = data.dropna()  # Drop missing values

# Drop non-numeric columns (like 'Product type' and 'SKU') before scaling
data = data.drop(columns=['Product type', 'SKU'], errors='ignore')

# Convert Date to datetime (if applicable)
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    # Feature engineering from Date
    data['Month'] = data['Date'].dt.month
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Quarter'] = data['Date'].dt.quarter
    data.drop(columns=['Date'], inplace=True)

# Step 4: One-hot Encoding Categorical Columns
# List of categorical columns to be encoded
categorical_cols = ['Promotion', 'Weather', 'EconomicIndicators', 'Customer demographics', 'Shipping carriers', 'Supplier name', 'Location', 'Transportation modes', 'Routes', 'Inspection results', 'Defect rates']
for col in categorical_cols:
    if col in data.columns:  # Ensure the column exists in the DataFrame
        # Check if column contains 'Fail' or other strings, use LabelEncoder or One-Hot Encoding
        if data[col].dtype == 'object':  # Categorical columns
            data[col] = data[col].astype('category').cat.codes  # Label encoding (or use pd.get_dummies for one-hot encoding)

# Step 5: Define Features and Target
# Use 'Number of products sold' or other appropriate column as the target variable
target_column = 'Number of products sold'  # Update this if needed
if target_column not in data.columns:
    raise ValueError(f"The dataset must contain '{target_column}' as the target column.")

X = data.drop(columns=[target_column])
y = data[target_column]

# Step 6: Train-Test Split and Scaling
print("\nSplitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure all data passed to StandardScaler is numeric
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Build Neural Network Model
print("\nBuilding model...")
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Step 8: Train the Model
print("\nTraining model...")
history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, verbose=1)

# Step 9: Plot Training and Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Model Training History')
plt.legend()
plt.tight_layout()
plt.show()

# Step 10: Evaluate the Model
print("\nEvaluating model...")
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Scatter Plot: True vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True Sales")
plt.ylabel("Predicted Sales")
plt.title("True vs Predicted Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: Save the Trained Model
print("\nSaving the model as 'demand_forecasting_model.h5'...")
model.save('demand_forecasting_model.h5')

# Step 12: Predict with Example Data
print("\nExample prediction using dummy input:")
example_input = np.random.rand(1, X.shape[1])
example_input_scaled = scaler.transform(example_input)
example_prediction = model.predict(example_input_scaled)
print(f"Predicted Sales: {example_prediction[0][0]:.2f}")
