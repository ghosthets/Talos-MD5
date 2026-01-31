# scripts/create_dummy_data.py

import pandas as pd
import json
import random
from datetime import datetime, timedelta

def generate_dummy_data(num_samples=1000):
    """
    Dummy training data generate karo
    """
    
    data = []
    
    # Attack patterns
    sqli_payloads = [
        "' OR '1'='1",
        "admin' --",
        "1' UNION SELECT NULL--",
        "'; DROP TABLE users--"
    ]
    
    xss_payloads = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert('XSS')"
    ]
    
    normal_payloads = [
        "search=laptop",
        "user=john&password=secret123",
        "page=1&sort=desc"
    ]
    
    for i in range(num_samples):
        # Random attack type
        attack_type = random.choice(['benign', 'sqli', 'xss', 'benign', 'benign'])
        
        if attack_type == 'sqli':
            payload = random.choice(sqli_payloads)
            label = 'sqli'
            crs_score = random.randint(10, 20)
        elif attack_type == 'xss':
            payload = random.choice(xss_payloads)
            label = 'xss'
            crs_score = random.randint(8, 15)
        else:
            payload = random.choice(normal_payloads)
            label = 'benign'
            crs_score = random.randint(0, 3)
        
        # Create request JSON
        request = {
            'request_id': f'req_{i}',
            'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            'http_metadata': {
                'method': random.choice(['GET', 'POST', 'POST', 'GET']),
                'uri': f'/api/{random.choice(["login", "search", "user", "product"])}',
                'user_agent': 'Mozilla/5.0',
                'content_length': len(payload)
            },
            'request_data': {
                'query_string': payload if random.random() > 0.5 else '',
                'post_body': payload if random.random() > 0.5 else '',
                'cookies': 'session=abc123'
            },
            'crs_results': {
                'total_score': crs_score,
                'rules_triggered': [
                    {
                        'rule_id': '942100',
                        'category': label if label != 'benign' else 'protocol',
                        'severity': 'critical' if crs_score > 10 else 'medium',
                        'score': crs_score
                    }
                ] if crs_score > 5 else []
            }
        }
        
        data.append({
            'request_data': json.dumps(request),
            'label': label
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save
    df.to_csv('data/training_logs.csv', index=False)
    print(f"✅ Generated {num_samples} dummy samples!")
    print(f"Label distribution:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    generate_dummy_data(1000)