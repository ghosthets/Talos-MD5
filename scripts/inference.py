import joblib
import json
import sys
import os
import io
import numpy as np

# Import feature extractor
sys.path.append(os.path.dirname(__file__))
from feature_extractor import FeatureExtractor

# Windows Encoding Fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class WorldClassWAF:
    def __init__(self, models_dir='models'):
        print("🚀 Initializing MD5 Sentinel Engine...")
        try:
            # Loading models with metadata
            rf_data = joblib.load(f'{models_dir}/randomforest_latest.pkl')
            self.rf_model = rf_data['model'] if isinstance(rf_data, dict) else rf_data
            
            self.if_model = joblib.load(f'{models_dir}/isolationforest_latest.pkl')
            
            preprocessor = joblib.load(f'{models_dir}/preprocessor_latest.pkl')
            self.scaler = preprocessor['scaler']
            self.label_encoder = preprocessor['label_encoder']
            self.feature_names = preprocessor['feature_names']
            
            self.extractor = FeatureExtractor()
            print("✅ Core Brain Loaded: Advanced Intelligence Online\n")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Brain Damage (Models Missing): {str(e)}")
            sys.exit(1)
    
    def _calculate_threat_score(self, rf_conf, rf_class, if_score, crs_score):
        """World-class scoring algorithm: Calculates 0-100 risk score"""
        score = 0
        # 1. CRS Impact (Weight: 40%)
        score += min(crs_score * 4, 40)
        
        # 2. ML Classification Impact (Weight: 40%)
        if rf_class != 'benign':
            score += (rf_conf * 40)
        
        # 3. Anomaly Impact (Weight: 20%)
        # Normalizing IF score (usually between -0.8 and -0.4 for anomalies)
        if if_score < -0.5:
            anomaly_weight = abs(if_score) * 20
            score += min(anomaly_weight, 20)
            
        return float(min(round(score, 2), 100.0))

    def predict(self, request_json):
        if isinstance(request_json, list):
            request_json = request_json[0]

        req_data = request_json.get('request_data', request_json)
        
        # 1. Advanced Feature Extraction
        features = self.extractor.extract_features(req_data)
        
        # 2. Method Encoding
        metadata = req_data.get('http_metadata', {})
        method = metadata.get('method', 'GET').upper()
        method_map = {'GET': 0, 'POST': 1, 'PUT': 2, 'DELETE': 3}
        features['method_encoded'] = method_map.get(method, 0)
        
        # 3. Alignment & Scaling
        try:
            feature_array = [features.get(fname, 0) for fname in self.feature_names]
            feature_scaled = self.scaler.transform([feature_array])
        except Exception as e:
            return {"error": f"Intelligence Mismatch: {str(e)}"}
        
        # 4. Multi-Model Analysis
        # RandomForest: Signature/Pattern recognition
        rf_proba = self.rf_model.predict_proba(feature_scaled)[0]
        rf_pred = np.argmax(rf_proba)
        rf_class = self.label_encoder.inverse_transform([rf_pred])[0]
        rf_conf = float(rf_proba[rf_pred])
        
        # IsolationForest: Zero-day/Strange behavior detection
        if_score = float(self.if_model.score_samples(feature_scaled)[0])
        is_anomaly = bool(if_score < -0.55) # Advanced threshold

        # 5. Global Threat Scoring
        crs_score = int(request_json.get('crs_results', {}).get('total_score', 0))
        threat_score = self._calculate_threat_score(rf_conf, rf_class, if_score, crs_score)

        # 6. Final World-Class Decision Logic
        decision = self._make_final_verdict(threat_score, rf_class, rf_conf, is_anomaly, crs_score)
        
        return {
            'request_id': str(request_json.get('request_id', 'unknown')),
            'verdict': decision,
            'threat_intelligence': {
                'risk_score': threat_score,
                'threat_level': self._get_level(threat_score),
                'ml_insights': {
                    'classification': rf_class,
                    'certainty': round(rf_conf * 100, 2),
                    'anomaly_detected': is_anomaly,
                    'behavioral_score': round(if_score, 4)
                },
                'signature_score': crs_score
            }
        }

    def _get_level(self, score):
        if score < 15: return "CLEAN"
        if score < 40: return "SUSPICIOUS"
        if score < 70: return "HIGH RISK"
        return "CRITICAL"

    def _make_final_verdict(self, threat_score, rf_class, rf_conf, is_anomaly, crs_score):
        # Master Decision Matrix
        if threat_score >= 70 or crs_score >= 15:
            return {'action': 'BLOCK', 'reason': f'High Threat Detected ({rf_class})', 'severity': 'CRITICAL'}
        
        if threat_score >= 35:
            return {'action': 'CHALLENGE', 'reason': 'Anomalous Request Patterns', 'severity': 'MEDIUM'}
            
        if is_anomaly and rf_class == 'benign':
             return {'action': 'MONITOR', 'reason': 'Unknown pattern but suspicious behavior', 'severity': 'LOW'}

        return {'action': 'ALLOW', 'reason': 'Safe request verified', 'severity': 'INFO'}

def main():
    if len(sys.argv) < 2:
        print("🚀 MD5 Sentinel: Usage - python inference.py <file.json>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        request = json.load(f)
    
    waf = WorldClassWAF()
    
    if isinstance(request, list):
        print(f"📦 Analyzing Batch: {len(request)} requests...")
        result = [waf.predict(r) for r in request]
    else:
        result = waf.predict(request)
    
    print("\n" + "═"*60)
    print("🛡️  MD5 SENTINEL - ADVANCED THREAT ANALYSIS REPORT")
    print("═"*60)
    print(json.dumps(result, indent=2))
    print("═"*60)

if __name__ == "__main__":
    main()