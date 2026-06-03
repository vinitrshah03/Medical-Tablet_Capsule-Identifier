import re

def parse_drug_details(raw_name):
    """Parses the NIH string into specific drug components."""
    details = {
        "full_name": raw_name,
        "chemical_name": "Unknown",
        "dosage": "Unknown",
        "form": "Unknown",
        "brand": "Unknown Brand" # Changed from brand_name to brand to match app.py
    }

    # Extract Brand
    if "[" in raw_name and "]" in raw_name:
        details["brand"] = raw_name.split("[")[1].split("]")[0]

    # Extract Dosage using robust Regex
    dosage_match = re.search(r'(\d+\.?\d*)\s*(MG|ML|MCG)', raw_name, re.IGNORECASE)
    if dosage_match:
        details["dosage"] = f"{dosage_match.group(1)} {dosage_match.group(2)}"
            
    # Extract Chemical
    if details["dosage"] != "Unknown":
        parts = raw_name.split(details["dosage"])
        details["chemical_name"] = parts[0].strip()

    # Identify Form
    if "capsule" in raw_name.lower(): details["form"] = "Capsule"
    elif "tablet" in raw_name.lower(): details["form"] = "Tablet"

    return details

def re_rank_results(temp_results, ocr_text):
    """
    Implements Decision-Level Fusion by boosting candidates 
    that match the extracted text.
    """
    ocr_blob = ocr_text.lower()
    
    for item in temp_results:
        match_score = 0
        
        # 1. Dosage Match: Does the OCR see the number in the NIH dosage?
        dosage_nums = re.findall(r'\d+', item["dosage"])
        if dosage_nums and any(num in ocr_blob for num in dosage_nums):
            match_score += 150 
            item["verified"] = True
        else:
            item["verified"] = False
            
        item["adjusted_score"] = item["confidence"] + match_score

    # Sort by the new score to promote the correct pill
    return sorted(temp_results, key=lambda x: x['adjusted_score'], reverse=True)