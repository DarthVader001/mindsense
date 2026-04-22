"""
audio_model.py
Tries to use trained ML model (audio_classifier.pkl) if available.
Falls back to rule-based Librosa analysis if model not yet trained.
"""

import os
import pickle
import numpy as np

MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'audio_classifier.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'audio_label_encoder.pkl')

RISK_SCORES = {
    'depression': 75,
    'anxiety':    65,
    'stress':     50,
    'neutral':    20,
    'positive':   10,
}


def extract_features(file_path):
    """Extract 32 Librosa features from audio file."""
    import librosa
    y, sr = librosa.load(file_path, sr=22050, duration=30)
    features = []

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1).tolist())
    features.extend(np.std(mfccs,  axis=1).tolist())

    f0 = librosa.yin(y, fmin=50, fmax=500)
    f0_voiced = f0[f0 > 0]
    features.append(float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0)
    features.append(float(np.std(f0_voiced))  if len(f0_voiced) > 0 else 0)

    rms = librosa.feature.rms(y=y)[0]
    features.append(float(np.mean(rms)))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(np.asarray(tempo).flat[0]))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(float(np.mean(centroid)))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.append(float(np.mean(zcr)))

    return np.array(features), y, sr


def analyze_audio(audio_path):
    try:
        import librosa

        feats, y, sr = extract_features(audio_path)

        # ── Try ML model first ─────────────────────────────
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            with open(MODEL_PATH,   'rb') as f: clf = pickle.load(f)
            with open(ENCODER_PATH, 'rb') as f: le  = pickle.load(f)

            pred_enc  = clf.predict([feats])[0]
            pred_prob = clf.predict_proba([feats])[0]
            label     = le.inverse_transform([pred_enc])[0]
            confidence = round(max(pred_prob) * 100, 1)
            risk_score = RISK_SCORES.get(label, 30)

            return {
                'method':     'ML Model (Random Forest)',
                'label':      label,
                'confidence': confidence,
                'risk_score': risk_score,
                'features': {
                    'mean_pitch_hz':    round(feats[26], 1),
                    'pitch_variability':round(feats[27], 1),
                    'mean_energy':      round(feats[28], 4),
                    'tempo_bpm':        round(feats[29], 1),
                    'spectral_centroid':round(feats[30], 1),
                },
                'audio_duration_sec': round(len(y) / sr, 1)
            }

        # ── Fallback: rule-based ───────────────────────────
        mean_pitch  = feats[26]
        pitch_std   = feats[27]
        mean_energy = feats[28]
        tempo       = feats[29]

        risk_score = 0
        indicators = []

        if mean_pitch > 0:
            if mean_pitch < 100:
                risk_score += 25
                indicators.append('Unusually low pitch')
            elif mean_pitch < 130:
                risk_score += 10
                indicators.append('Below-average pitch')

        if pitch_std < 15:
            risk_score += 20
            indicators.append('Monotone speech')

        if mean_energy < 0.01:
            risk_score += 20
            indicators.append('Low speech energy')

        if tempo < 60:
            risk_score += 15
            indicators.append('Slow speech rate')

        risk_score = min(risk_score, 100)
        if not indicators:
            indicators.append('No significant markers detected')

        return {
            'method':     'Rule-based (train model for ML)',
            'risk_score': round(risk_score, 1),
            'features': {
                'mean_pitch_hz':    round(mean_pitch, 1),
                'pitch_variability':round(pitch_std, 1),
                'mean_energy':      round(mean_energy, 4),
                'tempo_bpm':        round(tempo, 1),
                'spectral_centroid':round(feats[30], 1),
            },
            'indicators': indicators,
            'audio_duration_sec': round(len(y) / sr, 1)
        }

    except ImportError:
        return {'risk_score': 0, 'error': 'librosa not installed. Run: pip install librosa'}
    except Exception as e:
        return {'risk_score': 0, 'error': str(e)}
