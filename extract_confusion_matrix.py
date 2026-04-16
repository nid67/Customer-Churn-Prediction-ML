"""
Extract confusion matrices from churn prediction models
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project directory to path
sys.path.insert(0, r'C:\Users\Nidhin\OneDrive\Documents\College\ML MINI PROJECT')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_churn_dataset(filepath):
    """Load real Telco Customer Churn dataset"""
    df = pd.read_csv(filepath)
    
    # Handle TotalCharges - convert to numeric, replace empty strings with NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    
    # Convert Churn to binary
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Drop customerID as it's not a feature
    df = df.drop('customerID', axis=1)
    
    return df

def preprocess_data(df):
    """Preprocess Telco dataset - encode categorical variables"""
    df_processed = df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in numeric_cols:
        numeric_cols.remove('Churn')
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical variables
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    return df_processed

def handle_class_imbalance(X_train, y_train):
    """Apply SMOTE for class imbalance"""
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced

def train_models(X_train, y_train, X_test, y_test):
    """Train LR and RF models"""
    models = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    return models

def extract_confusion_matrices(models, X_test, y_test):
    """Extract confusion matrix values for all models"""
    confusion_matrices = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        confusion_matrices[model_name] = {
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp
        }
    
    return confusion_matrices

# Main execution
if __name__ == "__main__":
    try:
        print("Loading dataset...")
        dataset_path = r"C:\Users\Nidhin\OneDrive\Documents\College\ML MINI PROJECT\Telco-Customer-Churn.csv"
        df = load_churn_dataset(dataset_path)
        
        print("Preprocessing data...")
        df_processed = preprocess_data(df)
        X = df_processed.drop('Churn', axis=1)
        y = df_processed['Churn']
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("Handling class imbalance...")
        X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
        
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        print("Training models...")
        models = train_models(X_train_scaled, y_train_balanced, X_test_scaled, y_test)
        
        print("Extracting confusion matrices...\n")
        confusion_matrices = extract_confusion_matrices(models, X_test_scaled, y_test)
        
        # Display confusion matrices
        print("=" * 50)
        print("CONFUSION MATRIX VALUES")
        print("=" * 50)
        
        for model_name in ['Logistic Regression', 'Random Forest']:
            if model_name in confusion_matrices:
                cm = confusion_matrices[model_name]
                print(f"\n{model_name}:")
                print(f"TN = {cm['TN']}")
                print(f"FP = {cm['FP']}")
                print(f"FN = {cm['FN']}")
                print(f"TP = {cm['TP']}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
