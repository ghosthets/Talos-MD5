import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from datetime import datetime
import sys
import os
import argparse
import io

# --- WINDOWS EMOJI & ENCODING FIX ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import feature extractor
sys.path.append(os.path.dirname(__file__))
from feature_extractor import FeatureExtractor

def train_models():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to training data')
    args = parser.parse_args()

    print("="*60)
    print("🚀 HYBRID WAF ML TRAINING PIPELINE")
    print("="*60)
    
    data_path = args.data if args.data else 'data/training_data.json'
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: File not found at {data_path}")
        return

    # 1. Load Data
    try:
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            df = pd.DataFrame(raw_data)
        else:
            df = pd.read_csv(data_path)
            
        print(f"✅ Loaded {len(df)} samples from {os.path.basename(data_path)}")
    except Exception as e:
        print(f"❌ ERROR loading file: {str(e)}")
        return

    # 2. Extract features
    print("[2/6] Extracting features...")
    extractor = FeatureExtractor()
    features_list = []
    labels = []
    
    for idx, row in df.iterrows():
        try:
            req_data = row['request_data']
            if isinstance(req_data, str):
                req_json = json.loads(req_data)
            else:
                req_json = req_data

            features = extractor.extract_features(req_json)
            
            # Manual Fix for nested 'method'
            if 'method' not in features:
                features['method'] = req_json.get('http_metadata', {}).get('method', 'GET')
            
            features_list.append(features)
            labels.append(row['label'])
            
        except Exception as e:
            continue 
            
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            print(f"Processed {idx + 1}/{len(df)} samples...")
    
    features_df = pd.DataFrame(features_list)
    
    if features_df.empty:
        print("❌ ERROR: No features extracted! Check JSON structure.")
        return

    # 3. Prepare data
    print(f"✅ Extracted {len(features_df.columns)} features")
    print("\n[3/6] Preparing data for training...")
    
    # Method encoding safely
    if 'method' in features_df.columns:
        method_map = {'GET': 0, 'POST': 1, 'PUT': 2, 'DELETE': 3}
        features_df['method_encoded'] = features_df['method'].map(method_map).fillna(0)
        features_df = features_df.drop(['method'], axis=1)
    
    features_df = features_df.fillna(0)
    X = features_df.values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # --- SMART BALANCING & SPLIT ---
    # First split raw data to avoid NameError
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use SMOTE only if we have enough samples for k_neighbors
    try:
        from imblearn.over_sampling import SMOTE
        min_samples = pd.Series(y_train).value_counts().min()
        
        if min_samples > 1:
            print("Applying SMOTE for class balance...")
            k = min(1, min_samples - 1)
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            print("⚠️ Skipping SMOTE: Not enough samples for minority class.")
    except Exception as e:
        print(f"⚠️ SMOTE skipped: {str(e)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train Models
    print("[4/6] Training RandomForest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    print("[5/6] Training IsolationForest...")
    # IsolationForest needs slightly different logic for small data
    if_model = IsolationForest(contamination=0.05, random_state=42)
    if_model.fit(X_train_scaled) 
    
    # 6. Save
    print("[6/6] Saving models...")
    os.makedirs('models', exist_ok=True)
    
    # Consistency check for saved models
    joblib.dump(rf_model, 'models/randomforest_latest.pkl')
    joblib.dump(if_model, 'models/isolationforest_latest.pkl')
    joblib.dump({
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': features_df.columns.tolist()
    }, 'models/preprocessor_latest.pkl')
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print(f"✅ Models saved in /models folder")
    print("="*60)

if __name__ == "__main__":
    train_models()