import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

def generate_drought_dataset():
    X, y = make_classification(
        n_samples=5000, 
        n_features=6, 
        n_informative=4, 
        n_redundant=1,
        n_classes=2, 
        weights=[0.8, 0.2], 
        random_state=42
    )
    
    feature_names = [
        'Monthly precipitation', 
        'Average temperature', 
        'Maximum temperature', 
        'Rainfall anomaly', 
        'Previous months\' rainfall',
        'Soil moisture index'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Drought_Target'] = y
    return df

def main():
    df = generate_drought_dataset()
    
    X = df[[
        'Monthly precipitation', 
        'Average temperature', 
        'Maximum temperature', 
        'Rainfall anomaly', 
        'Previous months\' rainfall',
        'Soil moisture index'
    ]]
    y = df['Drought_Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42, max_depth=6),
        "XGBoost Classifier": XGBClassifier(random_state=42, eval_metric='logloss', max_depth=5, learning_rate=0.1)
    }

    for name, model in models.items():
        print(f"\n" + "="*50)
        print(f"Training and Evaluating: {name}")
        print("="*50)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"-> Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
        print(f"-> Precision : {prec:.4f}")
        print(f"-> Recall    : {rec:.4f}")
        print(f"-> F1-Score  : {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()