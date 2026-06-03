import torch
import torch.nn as nn
from torchvision import models
import json
import os
import sys

# Add parent directory to path so we can import from project root if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def validate_model():
    print("--- Starting Model Integrity Check ---")
    
    MODEL_PATH = "model/pill_model.pth"
    MAPPING_PATH = "model/class_mapping.json"
    
    # 1. Check if files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH):
        print("ERROR: Model or Mapping file missing in 'model/' folder.")
        return False
    
    # 2. Load Mapping and get class count
    with open(MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    num_classes = len(mapping)
    print(f"Mapping loaded: {num_classes} classes found.")

    # 3. Initialize Model Architecture
    try:
        model = models.mobilenet_v3_large()
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        print(f"Architecture initialized with {num_classes} output neurons.")
    except Exception as e:
        print(f"ERROR: Failed to initialize architecture: {e}")
        return False

    # 4. Load State Dict (The most common point of failure)
    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print("Weights loaded successfully (Size Match Confirmed).")
    except RuntimeError as e:
        print(f"SIZE MISMATCH ERROR: Your .pth and .json are out of sync!\n{e}")
        return False
    except Exception as e:
        print(f"ERROR: Loading weights failed: {e}")
        return False

    # 5. Dummy Inference Test
    try:
        # Create a fake image tensor [Batch, Channel, Height, Width]
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        if output.shape[1] == num_classes:
            print(f"Inference Test Passed: Output shape {output.shape} matches mapping.")
        else:
            print(f"ERROR: Model output {output.shape[1]} doesn't match mapping {num_classes}.")
            return False
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        return False

    print("\nALL CHECKS PASSED: Your model is ready for the Flask Web UI!")
    return True

if __name__ == "__main__":
    validate_model()