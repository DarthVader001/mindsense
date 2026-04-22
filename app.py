import os, sys, json, threading
from flask import Flask, request, jsonify, render_template
from models.text_model import analyze_text
from models.questionnaire_model import analyze_questionnaire
from models.audio_model import analyze_audio

import gspread
from google.oauth2.service_account import Credentials

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Google Sheets setup ───────────────────────────────────
SPREADSHEET_ID = '1S6YnuO2wvF19lwDgRt7DsnZdRr53bD3703y2kp7I0-k'

def get_sheet():
    creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    if not creds_json:
        print("WARNING: GOOGLE_CREDENTIALS_JSON not set")
        return None
    creds_dict = json.loads(creds_json)
    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID).sheet1
    # Add header row if sheet is empty
    if sheet.row_count == 0 or sheet.cell(1, 1).value is None:
        sheet.append_row([
            'user_id', 'timestamp',
            'text_score', 'questionnaire_score', 'audio_score',
            'final_score', 'risk_level', 'modalities_used'
        ])
    return sheet

def log_to_sheets(user_id, results):
    """Run in background thread so it doesn't slow down the response"""
    try:
        import datetime
        sheet = get_sheet()
        if sheet is None:
            return
        fusion = results.get('fusion', {})
        row = [
            user_id,
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            results.get('text', {}).get('risk_score', ''),
            results.get('questionnaire', {}).get('risk_score', ''),
            results.get('audio', {}).get('risk_score', ''),
            fusion.get('final_score', ''),
            fusion.get('risk_level', ''),
            ', '.join(fusion.get('modalities_used', []))
        ]
        sheet.append_row(row)
        print(f"  Logged {user_id} to Google Sheets")
    except Exception as e:
        print(f"  Sheets logging error: {e}")

user_counter = 0


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    global user_counter
    results = {}
    modality_scores = {}

    # ── Modality 1: Text ──────────────────────────────────
    text_input = request.form.get('journal_text', '')
    if text_input.strip():
        text_result = analyze_text(text_input)
        results['text'] = text_result
        modality_scores['text'] = text_result['risk_score']

    # ── Modality 2: Questionnaire ─────────────────────────
    questionnaire_data = request.form.get('questionnaire', '')
    if questionnaire_data:
        try:
            answers  = json.loads(questionnaire_data)
            q_result = analyze_questionnaire(answers)
            results['questionnaire'] = q_result
            if 'risk_score' in q_result:
                modality_scores['questionnaire'] = q_result['risk_score']
        except Exception as e:
            results['questionnaire'] = {'error': str(e)}

    # ── Modality 3: Audio ─────────────────────────────────
    audio_file = request.files.get('audio_file')
    if audio_file and audio_file.filename:
        audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(audio_path)
        audio_result = analyze_audio(audio_path)
        results['audio'] = audio_result
        if audio_result.get('risk_score', 0) > 0:
            modality_scores['audio'] = audio_result['risk_score']
        try: os.remove(audio_path)
        except: pass
    else:
        results['audio'] = {'status': 'not_provided', 'message': 'No audio uploaded'}

    # ── Weighted Fusion ───────────────────────────────────
    WEIGHTS = {'questionnaire': 0.45, 'text': 0.35, 'audio': 0.20}

    if modality_scores:
        total_weight = sum(WEIGHTS[m] for m in modality_scores)
        final_score  = round(
            sum(modality_scores[m] * WEIGHTS[m] for m in modality_scores) / total_weight, 1
        )

        if final_score < 30:
            risk_level     = 'Low'
            recommendation = 'Your responses suggest low risk. Keep maintaining healthy habits and a support network.'
        elif final_score < 60:
            risk_level     = 'Moderate'
            recommendation = 'Moderate risk detected. Consider speaking with a counselor or a trusted person.'
        else:
            risk_level     = 'High'
            recommendation = 'High risk detected. Please reach out to a mental health professional as soon as possible.'

        results['fusion'] = {
            'final_score':     final_score,
            'risk_level':      risk_level,
            'recommendation':  recommendation,
            'modalities_used': list(modality_scores.keys()),
            'weights_applied': {m: WEIGHTS[m] for m in modality_scores},
            'note':            f"Score based on: {', '.join(modality_scores.keys())}"
        }

        # ── Log to Google Sheets in background ───────────
        user_counter += 1
        user_id = f"user_{user_counter:03d}"
        thread = threading.Thread(target=log_to_sheets, args=(user_id, results))
        thread.daemon = True
        thread.start()

    return jsonify(results)


if __name__ == '__main__':
    print("=========================================")
    print("  MindSense — Final Build")
    print("  Google Sheets logging: ON")
    print("  Open: http://localhost:5000")
    print("=========================================")
    app.run(debug=True, port=5000)
