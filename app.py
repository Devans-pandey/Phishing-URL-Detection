"""
Phishing URL Detection - Modern Multi-Model Flask Web Application
Active 4 Models: Logistic Regression, XGBoost, Decision Tree, SVM.
Other suite models (e.g. HistGradientBoosting, Random Forest) remain preserved in the background.
"""

import os
import pickle
import warnings
import numpy as np
from flask import Flask, request, render_template, jsonify
from feature import FeatureExtraction

warnings.filterwarnings('ignore')

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "pickle")
SUITE_PATH = os.path.join(MODELS_DIR, "models_suite.pkl")
SINGLE_MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")

# Active 4 models exposed in UI and ensemble calculations
ACTIVE_MODELS = [
    "Logistic Regression",
    "XGBoost",
    "Decision Tree",
    "SVM"
]

models_suite = {}
try:
    if os.path.exists(SUITE_PATH):
        with open(SUITE_PATH, "rb") as f:
            models_suite = pickle.load(f)
        print(f"Loaded {len(models_suite)} models in suite: {list(models_suite.keys())}")
    elif os.path.exists(SINGLE_MODEL_PATH):
        with open(SINGLE_MODEL_PATH, "rb") as f:
            single = pickle.load(f)
            models_suite['XGBoost'] = {'model': single, 'accuracy': 100.0}
except Exception as e:
    print(f"Warning: Could not load model suite: {e}")

def get_visible_models(include_hidden=False):
    """
    Returns visible models.
    By default, filters to only the 4 specified models:
    Logistic Regression, XGBoost, Decision Tree, SVM.
    Other models remain in models_suite but are hidden from the active display.
    """
    if include_hidden:
        return models_suite

    filtered = {}
    for name in ACTIVE_MODELS:
        if name in models_suite:
            filtered[name] = models_suite[name]
        else:
            alt_key = name.replace(" ", "")
            if alt_key in models_suite:
                filtered[name] = models_suite[alt_key]

    return filtered if filtered else models_suite

def analyze_url(url_string, active_model_name=None, include_hidden=False):
    if not url_string:
        return None

    url_cleaned = url_string.strip()
    if not url_cleaned.startswith(('http://', 'https://')):
        url_cleaned = 'http://' + url_cleaned

    obj = FeatureExtraction(url_cleaned)
    features_df = obj.getFeaturesDataFrame()

    visible_suite = get_visible_models(include_hidden=include_hidden)
    default_ensemble_name = f"Ensemble ({len(visible_suite)} Models)"
    if not active_model_name:
        active_model_name = default_ensemble_name

    comparison_results = []
    safe_votes = 0
    total_votes = 0

    # Evaluate across the active visible models
    for name, item in visible_suite.items():
        m = item['model']
        acc = item.get('accuracy', 100.0)
        try:
            pred = int(m.predict(features_df)[0])
            proba = m.predict_proba(features_df)[0]
            classes = list(m.classes_)
            phish_idx = classes.index(0) if 0 in classes else 0
            safe_idx = classes.index(1) if 1 in classes else 1

            p_phish = float(proba[phish_idx])
            p_safe = float(proba[safe_idx])
        except Exception:
            pred = 1
            p_safe, p_phish = 0.5, 0.5

        m_is_safe = (pred == 1)
        if m_is_safe:
            safe_votes += 1
        total_votes += 1

        conf = p_safe if m_is_safe else p_phish
        comparison_results.append({
            "name": name,
            "is_safe": m_is_safe,
            "verdict": "Safe" if m_is_safe else "Phishing",
            "confidence": round(conf * 100, 1),
            "prob_safe": round(p_safe * 100, 1),
            "prob_phishing": round(p_phish * 100, 1),
            "accuracy": acc
        })

    # Overall Consensus or Single Model Selection
    if active_model_name in visible_suite:
        target_res = next((r for r in comparison_results if r['name'] == active_model_name), comparison_results[0])
        overall_safe = bool(target_res['is_safe'])
        overall_conf = float(target_res['confidence'])
        overall_prob_safe = float(target_res['prob_safe'])
        overall_prob_phish = float(target_res['prob_phishing'])
    else:
        overall_safe = bool(safe_votes >= (total_votes / 2.0))
        if comparison_results:
            if overall_safe:
                overall_conf = float(round(float(np.mean([r['prob_safe'] for r in comparison_results])), 1))
            else:
                overall_conf = float(round(float(np.mean([r['prob_phishing'] for r in comparison_results])), 1))
            overall_prob_safe = float(round(float(np.mean([r['prob_safe'] for r in comparison_results])), 1))
            overall_prob_phish = float(round(float(np.mean([r['prob_phishing'] for r in comparison_results])), 1))
        else:
            overall_conf = 50.0
            overall_prob_safe = 50.0
            overall_prob_phish = 50.0

    feature_breakdown = []
    for item in obj.getFeaturesDict():
        val = item['value']
        if isinstance(val, (np.floating, float)):
            clean_val = round(float(val), 4)
        elif isinstance(val, (np.integer, int)):
            clean_val = int(val)
        else:
            clean_val = str(val)
        feature_breakdown.append({
            "name": item['name'],
            "status": item['status'],
            "score": clean_val
        })

    available_model_names = [default_ensemble_name] + list(visible_suite.keys())

    return {
        "url": url_cleaned,
        "is_safe": overall_safe,
        "confidence": overall_conf,
        "prob_safe": overall_prob_safe,
        "prob_phishing": overall_prob_phish,
        "domain": obj.domain,
        "active_model": active_model_name,
        "available_models": available_model_names,
        "comparison": comparison_results,
        "features": feature_breakdown
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    show_all = request.args.get("show_all") == "1" or request.form.get("show_all") == "1"
    visible_suite = get_visible_models(include_hidden=show_all)
    default_ensemble_label = f"Ensemble ({len(visible_suite)} Models)"
    available_models = [default_ensemble_label] + list(visible_suite.keys())

    if request.method == "POST":
        input_url = request.form.get("url", "").strip()
        selected_model = request.form.get("model_choice", default_ensemble_label)
        if input_url:
            result = analyze_url(input_url, selected_model, include_hidden=show_all)

    return render_template(
        "index.html",
        result=result,
        available_models=available_models,
        show_all=show_all,
        total_models_count=len(models_suite)
    )

@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(silent=True) or {}
    url = data.get("url") or request.form.get("url")
    show_all = bool(data.get("show_all") or request.form.get("show_all"))
    visible_suite = get_visible_models(include_hidden=show_all)
    default_label = f"Ensemble ({len(visible_suite)} Models)"
    model_choice = data.get("model") or default_label

    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    analysis = analyze_url(url, model_choice, include_hidden=show_all)
    return jsonify({
        "status": "success",
        "data": analysis
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f">> Phishing URL Detection Server running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)