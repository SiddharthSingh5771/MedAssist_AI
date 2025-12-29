import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import sys

# Get the current directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the 'saved_models' folder if it doesn't exist
save_path = os.path.join(current_dir, 'saved_models')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# --- 1. Train Diabetes Model ---
print("Training Diabetes Model...")
try:
    # Load dataset
    csv_path = os.path.join(current_dir, 'diabetes.csv')
    diabetes_df = pd.read_csv(csv_path)
    
    X_diab = diabetes_df.drop(columns='Outcome', axis=1)
    Y_diab = diabetes_df['Outcome']
    
    diab_model = LogisticRegression(max_iter=1000)
    diab_model.fit(X_diab, Y_diab)
    
    # Save model
    model_path = os.path.join(save_path, 'diabetes_model.sav')
    pickle.dump(diab_model, open(model_path, 'wb'))
    print(f"✅ Diabetes model saved to: {model_path}")
    
except FileNotFoundError:
    print("❌ Error: 'diabetes.csv' not found. Please make sure the csv file is in the same folder.")
except Exception as e:
    print(f"❌ An error occurred: {e}")

# --- 2. Train Heart Disease Model ---
print("\nTraining Heart Disease Model...")
try:
    # Load dataset
    csv_path = os.path.join(current_dir, 'heart.csv')
    heart_df = pd.read_csv(csv_path)
    
    X_heart = heart_df.drop(columns='target', axis=1)
    Y_heart = heart_df['target']
    
    heart_model = RandomForestClassifier()
    heart_model.fit(X_heart, Y_heart)
    
    # Save model
    model_path = os.path.join(save_path, 'heart_model.sav')
    pickle.dump(heart_model, open(model_path, 'wb'))
    print(f"✅ Heart disease model saved to: {model_path}")
    
except FileNotFoundError:
    print("❌ Error: 'heart.csv' not found. Please make sure the csv file is in the same folder.")
except Exception as e:
    print(f"❌ An error occurred: {e}")