import re
import math
from urllib.parse import unquote
import numpy as np

class FeatureExtractor:
    """
    HTTP request se features extract karta hai - Advanced Sentinel Edition
    """
    
    # Advanced Regex Patterns for World-Class Detection
    PATTERNS = {
        'sqli': r"(union\s+all\s+select|select\s+.*\s+from|insert\s+into|drop\s+table|sleep\(\d+\)|benchmark\(|group\s+by|order\s+by|--|#)",
        'xss': r"(<script|alert\(|onerror|onload|eval\(|javascript:|atob\(|String\.fromCharCode|document\.cookie|<svg|<iframe)",
        'lfi_rce': r"(\.\.\/|\.\.\\|/etc/passwd|/windows/system32|cmd\.exe|/bin/sh|powershell|curl\s|wget\s|python\s-c|nc\s-e)",
        'nosql': r"(\$gt|\$ne|\$lookup|\$where|\$regex)"
    }
    
    SQL_KEYWORDS = ['select', 'union', 'insert', 'update', 'delete', 'drop', 'alter', 'create', 'exec', 'execute', 'declare']
    COMMAND_KEYWORDS = ['bash', 'sh', 'curl', 'wget', 'nc', 'netcat', 'cmd', 'powershell', 'eval', 'system', 'exec']
    
    def extract_features(self, request_json):
        features = {}
        
        # 1. HTTP Metadata (Basic but essential)
        meta = request_json.get('http_metadata', {})
        features['uri_length'] = len(meta.get('uri', ''))
        features['content_length'] = int(meta.get('content_length', 0))
        
        # 2. Payload Preparation (Normalization is key!)
        req_data = request_json.get('request_data', {})
        query_string = unquote(str(req_data.get('query_string', ''))) # Decoding is vital
        post_body = unquote(str(req_data.get('post_body', '')))
        cookies = unquote(str(req_data.get('cookies', '')))
        
        full_payload = f"{query_string} {post_body} {cookies}"
        payload_lower = full_payload.lower()
        
        # 3. Statistical Intelligence (World-Class Features)
        features['len'] = len(full_payload)
        features['special_char_ratio'] = self._special_char_ratio(full_payload)
        features['entropy'] = self._calculate_entropy(full_payload)
        features['numeric_ratio'] = self._numeric_ratio(full_payload)
        features['uppercase_ratio'] = self._uppercase_ratio(full_payload)
        
        # Obfuscation Detection (Hackers use this to bypass WAF)
        features['encoded_char_count'] = full_payload.count('%')
        features['hex_detection'] = 1 if re.search(r'\\x[0-9a-fA-F]{2}', full_payload) else 0
        features['null_byte'] = 1 if '%00' in full_payload or '\x00' in full_payload else 0

        # 4. Pattern Intelligence (The "Sniper" Part)
        for name, pattern in self.PATTERNS.items():
            features[f'risk_{name}'] = len(re.findall(pattern, payload_lower))

        # Keyword Density
        features['sql_keyword_count'] = sum(1 for kw in self.SQL_KEYWORDS if kw in payload_lower)
        features['cmd_keyword_count'] = sum(1 for kw in self.COMMAND_KEYWORDS if kw in payload_lower)
        
        # 5. CRS/Rule Engine Synergy
        crs = request_json.get('crs_results', {})
        features['crs_total_score'] = crs.get('total_score', 0)
        features['crs_rules_hit'] = len(crs.get('rules_triggered', []))
        
        # Flattening Category Scores
        cat_scores = self._aggregate_crs_categories(crs.get('rules_triggered', []))
        features.update(cat_scores)

        return features
    
    def _special_char_ratio(self, text):
        if not text: return 0.0
        # Count dangerous chars mostly used in attacks
        dangerous_chars = len(re.findall(r'[<>\'\"();&|`$\\]', text))
        return dangerous_chars / len(text)
    
    def _calculate_entropy(self, text):
        if not text: return 0.0
        prob = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * math.log2(p) for p in prob if p > 0)
    
    def _numeric_ratio(self, text):
        return sum(c.isdigit() for c in text) / len(text) if text else 0.0

    def _uppercase_ratio(self, text):
        return sum(c.isupper() for c in text) / len(text) if text else 0.0
    
    def _aggregate_crs_categories(self, rules):
        scores = {f'crs_{c}_score': 0 for c in ['sqli', 'xss', 'lfi', 'rce']}
        for rule in rules:
            cat = rule.get('category', '').lower()
            sc = rule.get('score', 0)
            if 'sqli' in cat: scores['crs_sqli_score'] += sc
            elif 'xss' in cat: scores['crs_xss_score'] += sc
            elif 'lfi' in cat or 'traversal' in cat: scores['crs_lfi_score'] += sc
            elif 'rce' in cat or 'command' in cat: scores['crs_rce_score'] += sc
        return scores