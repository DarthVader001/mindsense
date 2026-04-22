"""
test_users.py — User Testing Logger
Run this alongside app.py. It logs every analysis result
from test users into test_results.csv for evaluation.

To use:
  1. Run app.py normally
  2. Also run this script: python test_users.py
     (it patches the /analyze endpoint to also log results)

OR — simpler approach: the logging is already built into app.py
if you use the --test flag: python app.py --test
"""

import csv
import os
from datetime import datetime

LOG_FILE = 'test_results.csv'
HEADERS  = [
    'timestamp', 'user_id', 'text_score', 'phq9_score', 'phq9_label',
    'gad7_score', 'gad7_label', 'audio_score', 'final_score',
    'risk_level', 'modalities_used'
]


def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()
        print(f"Created {LOG_FILE}")


def log_result(user_id, results):
    """Call this with the JSON results dict from /analyze."""
    init_log()
    row = {
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_id':        user_id,
        'text_score':     results.get('text', {}).get('risk_score', ''),
        'phq9_score':     results.get('questionnaire', {}).get('phq9', {}).get('total_score', ''),
        'phq9_label':     results.get('questionnaire', {}).get('phq9', {}).get('interpretation', ''),
        'gad7_score':     results.get('questionnaire', {}).get('gad7', {}).get('total_score', ''),
        'gad7_label':     results.get('questionnaire', {}).get('gad7', {}).get('interpretation', ''),
        'audio_score':    results.get('audio', {}).get('risk_score', ''),
        'final_score':    results.get('fusion', {}).get('final_score', ''),
        'risk_level':     results.get('fusion', {}).get('risk_level', ''),
        'modalities_used':','.join(results.get('fusion', {}).get('modalities_used', [])),
    }
    with open(LOG_FILE, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=HEADERS).writerow(row)


def print_summary():
    """Print summary statistics from logged test results."""
    if not os.path.exists(LOG_FILE):
        print("No test results yet.")
        return
    import pandas as pd
    df = pd.read_csv(LOG_FILE)
    print(f"\n{'='*40}")
    print(f"  TEST USER RESULTS SUMMARY")
    print(f"{'='*40}")
    print(f"Total users tested : {len(df)}")
    print(f"Avg final score    : {df['final_score'].mean():.1f}")
    print(f"\nRisk level distribution:")
    print(df['risk_level'].value_counts().to_string())
    print(f"\nAvg text score     : {df['text_score'].mean():.1f}")
    print(f"Avg PHQ-9 score    : {df['phq9_score'].mean():.1f}")
    print(f"{'='*40}\n")


if __name__ == '__main__':
    print_summary()
