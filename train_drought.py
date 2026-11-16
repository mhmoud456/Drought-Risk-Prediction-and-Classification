import numpy as np
import pandas as pd
import requests
from pathlib import Path
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

def load_climate_data(lat=33.5138, lon=36.2765, start_year=1981, end_year=2025):
    cache = Path(f"nasa_power_monthly_{lat}_{lon}_{start_year}_{end_year}.csv")
    if cache.exists():
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df

    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "T2M,T2M_MAX,PRECTOTCORR,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_year,
        "end": end_year,
        "format": "JSON"
    }
    
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["properties"]["parameter"]
    
    df = pd.DataFrame({
        "temp_mean": raw["T2M"],
        "temp_max": raw["T2M_MAX"],
        "precip": raw["PRECTOTCORR"],
        "humidity": raw["RH2M"]
    })
    
    df = df[df.index.astype(str).str.len() == 6]
    df.index = pd.to_datetime(df.index, format="%Y%m", errors='coerce')
    df = df.dropna(how='all')
    df.index.name = "date"
    
    df = df.replace(-999, np.nan)
    df.to_csv(cache)
    return df

def generate_drought_dataset():
    df = load_climate_data()
    df = df.dropna()
    
    precip_mean = df['precip'].mean()
    precip_std = df['precip'].std()
    
    df['Drought_Target'] = (df['precip'] < (precip_mean - 0.5 * precip_std)).astype(int)
    
    df['Rainfall_anomaly'] = df['precip'] - precip_mean
    df['Previous_months_rainfall'] = df['precip'].shift(1).fillna(precip_mean)
    
    return df

def main():
    print("[INFO] Loading real NASA POWER climate data...")
    df = generate_drought_dataset()
    
    features = [
        'precip', 
        'temp_mean', 
        'temp_max', 
        'humidity', 
        'Rainfall_anomaly', 
        'Previous_months_rainfall'
    ]
    
    X = df[features]
    y = df['Drought_Target']
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"[INFO] Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    models = {
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42, max_depth=5),
        "XGBoost Classifier": XGBClassifier(random_state=42, eval_metric='logloss', max_depth=4, learning_rate=0.1)
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