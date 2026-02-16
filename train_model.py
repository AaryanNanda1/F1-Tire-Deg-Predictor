import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import json

def train_model(data_df):
    """
    Trains a HistGradientBoostingRegressor model on the provided data.
    
    Args:
        data_df (pd.DataFrame): Processed data.
        
    Returns:
        HistGradientBoostingRegressor: Trained model.
    """
    # Separate features (X) and target variable (y)
    # We are predicting 'LapTimeSeconds' based on the other features.
    X = data_df.drop('LapTimeSeconds', axis=1)
    y = data_df['LapTimeSeconds']
    
    # Split data into training (80%) and testing (20%) sets.
    # This allows us to evaluate the model on data it hasn't seen before.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model: HistGradientBoostingRegressor
    # This efficiently handles large datasets and missing values natively.
    model = HistGradientBoostingRegressor(random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Evaluate model performance on the test set
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Model RMSE: {rmse}") # RMSE tells us the average error in seconds
    
    # Optional: Print feature usage (permutation importance is expensive, so we skip for speed, 
    # but we can print the number of features).
    print(f"Trained on {X.shape[1]} features.")
    
    # Save the trained model to a file
    joblib.dump(model, "tire_deg_model.joblib")
    print("Model saved to tire_deg_model.joblib")
    
    # Save the list of feature names.
    # CRITICAL: We need to ensure the order of columns is exactly the same during prediction.
    # We save as a list to avoid index issues.
    joblib.dump(X.columns.tolist(), 'model_features.joblib')
    print("Feature columns saved to model_features.joblib")

if __name__ == "__main__":
    from data_loader import load_race_data
    from preprocessing import preprocess_laps
    
    print("Loading data...")
    session = load_race_data(2023, 'Bahrain')
    data = preprocess_laps(session)
    
    print("Training model...")
    train_model(data)
