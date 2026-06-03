import time

from flask import Flask, render_template, request, url_for, redirect
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
import requests
from logic_utils import parse_drug_details, re_rank_results
from analytics_utils import generate_dashboard_charts
import csv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Shared memory for charts
PERFORMANCE_HISTORY = []
USER_FEEDBACK_LOG = []
FEEDBACK_FILE = "model/user_feedback.csv"

MODEL, MAPPING, OCR_READER = None, None, None

def load_resources():
    global MODEL, MAPPING, OCR_READER
    with open("model/class_mapping.json", 'r') as f:
        MAPPING = json.load(f)
    
    # Ensure this matches your train.py (Large vs Small)
    MODEL = models.mobilenet_v3_large()
    MODEL.classifier[3] = nn.Linear(MODEL.classifier[3].in_features, len(MAPPING))
    MODEL.load_state_dict(torch.load("model/pill_model.pth", map_location='cpu'))
    MODEL.eval()
    
    import easyocr
    OCR_READER = easyocr.Reader(['en'])
    print("Resources Loaded.")

def get_nih_data(result_code):
    ndc = result_code.split('_')[0]
    try:
        url = f"https://rxnav.nlm.nih.gov/REST/ndcstatus.json?ndc={ndc}"
        response = requests.get(url, timeout=5)
        name = response.json().get('ndcStatus', {}).get('conceptName')
        return name if name else f"Unknown Drug ({ndc})"
    except:
        return f"Connection Error ({ndc})"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction, confidence, image_url, analytics, ocr_log = [None]*5
    
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image_url = url_for('static', filename=f'uploads/{file.filename}')
            
            if not MODEL: load_resources()
            
            # 1. OCR
            ocr_result = OCR_READER.readtext(filepath, detail=0)
            ocr_log = " | ".join(ocr_result)
            
            # 2. CNN
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(Image.open(filepath).convert('RGB')).unsqueeze(0)
            
            with torch.no_grad():
                outputs = MODEL(img_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                top_probs, top_idxs = torch.topk(probs, 5)
                
                temp_results = []
                for i in range(5):
                    idx = top_idxs[i].item()
                    raw_code = MAPPING[str(idx)]
                    full_string = get_nih_data(raw_code)
                    details = parse_drug_details(full_string)
                    temp_results.append({
                        **details,
                        "confidence": round(top_probs[i].item() * 100, 2),
                        "raw_code": raw_code
                    })

                # 3. Re-Rank
                analytics = re_rank_results(temp_results, ocr_log)[:3]
                
                # 4. Log to History (Safety check added here)
                if analytics and len(analytics) > 0:
                    prediction = analytics[0]["full_name"]
                    confidence = analytics[0]["confidence"]
                    PERFORMANCE_HISTORY.append({
                        'label': analytics[0].get('brand', 'Unknown'), 
                        'expected_conf': 95, 
                        'actual_conf': analytics[0]['confidence']
                    })

    return render_template('index.html', prediction=prediction, confidence=confidence, 
                           image_url=image_url, analytics=analytics, ocr_text=ocr_log)

@app.route('/charts')
def charts():
    rmse_val = 0
    if len(PERFORMANCE_HISTORY) > 0:
        rmse_val = generate_dashboard_charts(PERFORMANCE_HISTORY, USER_FEEDBACK_LOG)
    return render_template('charts.html', rmse=round(rmse_val, 2))

# Ensure directory exists first
feedback_dir = os.path.dirname(FEEDBACK_FILE)
os.makedirs(feedback_dir, exist_ok=True)

# Ensure the CSV exists with headers
if not os.path.isfile(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'top_1_name', 'user_correct_indices', 'was_helpful'])

@app.route('/feedback', methods=['POST'])
def feedback():
    global USER_FEEDBACK_LOG
    data = request.form
    
    # Get indices of checkboxes (e.g., "0", "1", "2")
    correct_indices = request.form.getlist('correct_index')
    was_helpful = data.get('helpful') # "Yes" or "No"
    top_1_name = data.get('top_1_name')
    
    feedback_entry = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'top_1_name': top_1_name,
        'correct_indices': correct_indices,
        'was_helpful': was_helpful
    }
    
    # 1. Save to shared memory for current session charts
    USER_FEEDBACK_LOG.append(feedback_entry)
    
    # 2. Save to CSV for permanent dissertation records
    with open(FEEDBACK_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([feedback_entry['timestamp'], top_1_name, "|".join(correct_indices), was_helpful])
        
    return redirect(url_for('index'))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global PERFORMANCE_HISTORY, USER_FEEDBACK_LOG
    PERFORMANCE_HISTORY = []  # Reset the data list
    USER_FEEDBACK_LOG = []
    
    # Optional: Delete old charts so they don't show stale data
    chart_files = ['confidence_bar.png', 'rmse_chart.png', 'trend_chart.png', 
                   'residual_chart.png', 'confusion_heatmap.png']
    for f in chart_files:
        path = os.path.join('static/charts', f)
        if os.path.exists(path):
            os.remove(path)
            
    return redirect(url_for('charts'))

if __name__ == '__main__':
    load_resources()
    app.run(debug=True)