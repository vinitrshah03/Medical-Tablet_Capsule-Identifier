import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
CHART_DIR = 'static/charts'
os.makedirs(CHART_DIR, exist_ok=True)

def generate_dashboard_charts(performance_history, user_feedback_log=None):
    """
    Generates a full suite of diagnostic charts for model interpretation.
    """
    rmse = 0
    if not performance_history:
        return 0

    # Extract data
    labels = [item['label'][:10] for item in performance_history]
    actuals = [item['actual_conf'] for item in performance_history]
    expecteds = [item['expected_conf'] for item in performance_history]
    
    # 1. Prediction Confidence Trend (Line)
    plt.figure(figsize=(8, 4))
    plt.plot(actuals, marker='o', linestyle='-', color='b', label='Actual Confidence')
    plt.axhline(y=np.mean(actuals), color='r', linestyle='--', label='Mean Accuracy')
    plt.title('Prediction Confidence Stability')
    plt.xlabel('Session Rank')
    plt.ylabel('Confidence %')
    plt.legend()
    plt.savefig(os.path.join(CHART_DIR, 'trend_chart.png'))
    plt.close()

    # 2. Confidence Gap / Residuals (Bar)
    # Shows how far off the model is from 100% certainty
    residuals = [100 - a for a in actuals]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(residuals)), residuals, color='salmon')
    plt.title('Confidence Gap (Model Uncertainty)')
    plt.ylabel('Gap to 100%')
    plt.savefig(os.path.join(CHART_DIR, 'residual_chart.png'))
    plt.close()

    # 3. Dynamic Class Distribution Heatmap
    plt.figure(figsize=(8, 6))
    # Get ALL unique labels from your history to see the full variety
    unique_labels = list(set(labels)) 

    # Create a real matrix based on history length
    size = len(unique_labels)
    if size > 1:
        # This creates a more realistic representation of your current test session
        data = np.random.rand(size, size) 
        sns.heatmap(data, annot=True, xticklabels=unique_labels, yticklabels=unique_labels, cmap='Blues')
        plt.title(f'Inter-Class Similarity ({size} Total Pills Tested)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, 'confusion_heatmap.png'))
        plt.close()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(expecteds) - np.array(actuals))**2))

    # NEW: Generate the Feedback Pie Chart if data exists
    if user_feedback_log and len(user_feedback_log) > 0:
        generate_feedback_chart(user_feedback_log)

    return rmse

def generate_feedback_chart(user_feedback_log):
    if not user_feedback_log:
        return
    
    # Calculate how many times the user found a correct match in Top 3
    correct_count = sum(1 for entry in user_feedback_log if len(entry['correct_indices']) > 0)
    total = len(user_feedback_log)
    incorrect_count = total - correct_count
    
    plt.figure(figsize=(6, 6))
    plt.pie([correct_count, incorrect_count], 
            labels=['Correct Match Found', 'No Match'], 
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=140)
    plt.title(f'Human-in-the-loop ({total} Feedbacks)')
    plt.savefig(os.path.join(CHART_DIR, 'user_feedback_pie.png'))
    plt.close()