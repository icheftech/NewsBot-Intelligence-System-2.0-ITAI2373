"""NewsBot Intelligence System 2.0 - Utility Functions"""
import json
import os
from datetime import datetime
import pandas as pd

def save_results(data, filename, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_timestamp(dt=None):
    if dt is None:
        dt = datetime.now()
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def create_dataframe(articles_data):
    return pd.DataFrame(articles_data)

def export_to_csv(data, filename, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, encoding='utf-8')
    return filepath

def calculate_statistics(data_list):
    if not data_list:
        return {}
    return {'mean': sum(data_list) / len(data_list), 'min': min(data_list), 'max': max(data_list), 'count': len(data_list)}

def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result
