
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure",
    "Trouble concentrating on things",
    "Moving or speaking so slowly that others could have noticed",
    "Thoughts that you would be better off dead or of hurting yourself"
]

GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid, as if something awful might happen"
]


def interpret_phq9(score):
    if score <= 4:
        return 'Minimal or none', 'green'
    elif score <= 9:
        return 'Mild', 'yellow'
    elif score <= 14:
        return 'Moderate', 'orange'
    elif score <= 19:
        return 'Moderately severe', 'red'
    else:
        return 'Severe', 'darkred'


def interpret_gad7(score):
    if score <= 4:
        return 'Minimal anxiety', 'green'
    elif score <= 9:
        return 'Mild anxiety', 'yellow'
    elif score <= 14:
        return 'Moderate anxiety', 'orange'
    else:
        return 'Severe anxiety', 'darkred'


def analyze_questionnaire(answers):
    result = {}
    combined_scores = []

    # ── PHQ-9 ─────────────────────────────────────────────
    phq9_answers = answers.get('phq9', [])
    if phq9_answers and len(phq9_answers) == 9:
        phq9_total = sum(int(a) for a in phq9_answers)
        phq9_label, phq9_color = interpret_phq9(phq9_total)
        phq9_risk = round((phq9_total / 27) * 100, 1)

        result['phq9'] = {
            'status': 'complete',
            'total_score': phq9_total,
            'max_score': 27,
            'interpretation': phq9_label,
            'color': phq9_color,
            'flag_q9': int(phq9_answers[8]) > 0
        }
        combined_scores.append(phq9_risk)
    else:
        result['phq9'] = {'status': 'incomplete'}

    # ── GAD-7 ─────────────────────────────────────────────
    gad7_answers = answers.get('gad7', [])
    if gad7_answers and len(gad7_answers) == 7:
        gad7_total = sum(int(a) for a in gad7_answers)
        gad7_label, gad7_color = interpret_gad7(gad7_total)
        gad7_risk = round((gad7_total / 21) * 100, 1)

        result['gad7'] = {
            'status': 'complete',
            'total_score': gad7_total,
            'max_score': 21,
            'interpretation': gad7_label,
            'color': gad7_color
        }
        combined_scores.append(gad7_risk)
    else:
        result['gad7'] = {'status': 'incomplete'}

    # ── Combined questionnaire risk score ─────────────────
    if combined_scores:
        result['risk_score'] = round(sum(combined_scores) / len(combined_scores), 1)

    return result
