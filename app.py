import os, sys, json
from flask import Flask, request, jsonify, render_template
from models.text_model import analyze_text
from models.questionnaire_model import analyze_questionnaire
from models.audio_model import analyze_audio
from test_users import log_result, init_log

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pass --test flag to enable user logging: python app.py --test
TEST_MODE = '--test' in sys.argv
if TEST_MODE:
    init_log()
    print("  TEST MODE ON — results will be saved to test_results.csv")

user_counter = 0   # simple incrementing user ID


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
    # Questionnaire 45%  |  Text 35%  |  Audio 20%
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

        # ── Log test user result ──────────────────────────
        if TEST_MODE:
            user_counter += 1
            log_result(f"user_{user_counter:03d}", results)

    return jsonify(results)


if __name__ == '__main__':
    print("=========================================")
    print("  MindSense — Final Build")
    print("  Text NLP:          COMPLETE")
    print("  PHQ-9 + GAD-7:     COMPLETE")
    print("  Audio (Librosa):   COMPLETE")
    print("  Weighted Fusion:   COMPLETE")
    print(f"  Test logging:      {'ON' if TEST_MODE else 'OFF (use --test flag)'}")
    print("  Open: http://localhost:5000")
    print("=========================================")
    app.run(debug=True, port=5000)
