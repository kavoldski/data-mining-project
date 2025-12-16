# TMI4013 - Machine Learning for Business Analytics
# Group 5: Data Mining Project
# Group Members: Jesse, Cedric, Frank, Fauzi

import pandas as pd
import numpy as np
import os

# Scikit-learn modules for Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Scikit-learn modules for Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Scikit-learn modules for Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    print("--- STARTING TMI4013 PROJECT ---")

    # ======================================================
    # PART 1: DATA PREPROCESSING
    # ======================================================
    print("\n[Step 1] Loading and Preprocessing Data...")

    # 1. Load Data
    try:
        df_train = pd.read_excel('HR-Employee-Attrition.xlsx')
        df_predict = pd.read_excel('HR-Employee-Attrition-Predict.xlsx')
    except FileNotFoundError:
        print("ERROR: Excel files not found. Please make sure they are in the same folder as this script.")
        return

    # 2. Data Cleaning
    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df_train = df_train.drop(columns=drop_cols, errors='ignore')
    df_predict_clean = df_predict.drop(columns=drop_cols, errors='ignore')

    # 3. Encoding Target Variable (Attrition)
    le = LabelEncoder()
    if 'Attrition' in df_train.columns:
        df_train['Attrition'] = le.fit_transform(df_train['Attrition'])
        print(" - Target variable 'Attrition' encoded.")

    # 4. Feature Encoding (One-Hot Encoding)
    X = df_train.drop('Attrition', axis=1) # Features
    y = df_train['Attrition']              # Target

    X_encoded = pd.get_dummies(X, drop_first=True)

    # Process the Prediction file specifically to match Training columns
    X_predict_encoded = pd.get_dummies(df_predict_clean, drop_first=True)
    X_predict_final = X_predict_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_predict_scaled = scaler.transform(X_predict_final)

    # 6. Save Cleaned Dataset (Required Deliverable)
    clean_output = pd.DataFrame(X_scaled, columns=X_encoded.columns)
    clean_output['Attrition'] = y
    clean_output.to_csv('Cleaned_HR_Dataset.csv', index=False)
    print(" - Cleaned dataset saved as 'Cleaned_HR_Dataset.csv'")

    # ======================================================
    # PART 2: MACHINE LEARNING & OPTIMIZATION
    # ======================================================
    print("\n[Step 2] Training Models (this may take a moment)...")

    # Split Data (Holdout Method: 70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=2000),
            'params': {'C': [0.1, 1, 10]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {'C': [1, 10], 'kernel': ['rbf', 'linear']}
        }
    }

    results = []
    best_models = {}

    # GridSearchCV performs Cross-Validation to find best parameters
    for name, config in models_config.items():
        print(f" - Optimizing {name}...")
        clf = GridSearchCV(config['model'], config['params'], cv=3, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)
        
        best_models[name] = clf.best_estimator_
        
        # Predict on Test Set (Holdout)
        y_pred = clf.predict(X_test)
        
        # ======================================================
        # PART 3: RESULTS EVALUATION
        # ======================================================
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Algorithm': name,
            'Best Parameters': str(clf.best_params_),
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4)
        })

    # Display Comparison Table
    results_df = pd.DataFrame(results)
    print("\n[Step 3] Evaluation Results:")
    print(results_df.to_string())

    # Find and Display the Best Model
    best_idx = results_df['F1 Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Algorithm']
    best_model_f1 = results_df.loc[best_idx, 'F1 Score']
    best_model_acc = results_df.loc[best_idx, 'Accuracy']
    
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print("="*60)
    print(f"  Accuracy:  {best_model_acc}")
    print(f"  F1 Score:  {best_model_f1}")
    print("="*60)

    # ======================================================
    # PART 4: PREDICTION
    # ======================================================
    print("\n[Step 4] Predicting on New Samples...")

    # Get predictions from all 3 algorithms
    df_output = pd.read_excel('HR-Employee-Attrition-Predict.xlsx')
    
    all_predictions = {}
    for name, model in best_models.items():
        preds_numeric = model.predict(X_predict_scaled)
        preds_label = le.inverse_transform(preds_numeric)  # Back to Yes/No
        all_predictions[name] = preds_label
        df_output[f'Predicted_Attrition_{name.replace(" ", "_")}'] = preds_label

    # Display Predictions from All Algorithms
    print(" - Predictions from All Algorithms:")
    prediction_display = pd.DataFrame(all_predictions)
    print(prediction_display.to_string())
    
    # Use the best model for final prediction
    final_model = best_models[best_model_name]
    final_preds_numeric = final_model.predict(X_predict_scaled)
    final_preds_label = le.inverse_transform(final_preds_numeric)
    df_output['Predicted_Attrition'] = final_preds_label

    # Save Prediction Result
    df_output.to_excel('Final_Prediction_Results.xlsx', index=False)
    
    print(f"\n - Final Predictions (using {best_model_name}):")
    print(df_output[['Predicted_Attrition']])
    print(" - Prediction file saved as 'Final_Prediction_Results.xlsx'")
    print("\n--- PROJECT COMPLETE ---")

if __name__ == "__main__":
    main()