import requests
import os
import pandas as pd
import io

BASE_URL = "http://127.0.0.1:8000"

def print_status(component, status, message=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {component}: {message}")

def test_health():
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print_status("Health Check", True, "Server is reachable")
        else:
            print_status("Health Check", False, f"Status Code {response.status_code}")
    except Exception as e:
        print_status("Health Check", False, f"Connection Failed: {str(e)}")

def test_training():
    print("\n[ Testing Training Endpoint ]")
    try:
        # Note: This might take a while if actual training happens
        response = requests.post(f"{BASE_URL}/api/train")
        if response.status_code == 200:
            data = response.json()
            print_status("Training", True, data.get("message", "Training started"))
        else:
            print_status("Training", False, f"Failed with {response.status_code}: {response.text}")
    except Exception as e:
        print_status("Training", False, str(e))

def test_single_prediction():
    print("\n[ Testing Single Prediction ]")
    # Sample legitimate-looking data (mostly 1s or 0s depending on feature encoding)
    data = {
        "having_IP_Address": 1,
        "URL_Length": 1,
        "Shortining_Service": 1,
        "having_At_Symbol": 1,
        "double_slash_redirecting": 1,
        "Prefix_Suffix": 1,
        "having_Sub_Domain": 1,
        "SSLfinal_State": 1,
        "Domain_registeration_length": 1,
        "Favicon": 1,
        "port": 1,
        "HTTPS_token": 1,
        "Request_URL": 1,
        "URL_of_Anchor": 1,
        "Links_in_tags": 1,
        "SFH": 1,
        "Submitting_to_email": 1,
        "Abnormal_URL": 1,
        "Redirect": 1,
        "on_mouseover": 1,
        "RightClick": 1,
        "popUpWidnow": 1,
        "Iframe": 1,
        "age_of_domain": 1,
        "DNSRecord": 1,
        "web_traffic": 1,
        "Page_Rank": 1,
        "Google_Index": 1,
        "Links_pointing_to_page": 1,
        "Statistical_report": 1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/predict/single", data=data)
        if response.status_code == 200:
            result = response.json()
            print_status("Single Predict", True, f"Prediction: {result.get('prediction')} (Safe: {result.get('is_safe')})")
        else:
            print_status("Single Predict", False, f"Status {response.status_code}: {response.text}")
    except Exception as e:
        print_status("Single Predict", False, str(e))

def test_batch_prediction():
    print("\n[ Testing Batch Prediction ]")
    # Create dummy CSV
    df = pd.DataFrame([{
        "having_IP_Address": 1, "URL_Length": 1, "Shortining_Service": 1, "having_At_Symbol": 1,
        "double_slash_redirecting": 1, "Prefix_Suffix": 1, "having_Sub_Domain": 1, "SSLfinal_State": 1,
        "Domain_registeration_length": 1, "Favicon": 1, "port": 1, "HTTPS_token": 1, "Request_URL": 1,
        "URL_of_Anchor": 1, "Links_in_tags": 1, "SFH": 1, "Submitting_to_email": 1, "Abnormal_URL": 1,
        "Redirect": 1, "on_mouseover": 1, "RightClick": 1, "popUpWidnow": 1, "Iframe": 1, "age_of_domain": 1,
        "DNSRecord": 1, "web_traffic": 1, "Page_Rank": 1, "Google_Index": 1, "Links_pointing_to_page": 1,
        "Statistical_report": 1
    }] * 5) # 5 rows
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    files = {'file': ('test_batch.csv', csv_buffer.getvalue(), 'text/csv')}
    
    try:
        response = requests.post(f"{BASE_URL}/predict", files=files)
        if response.status_code == 200:
            # It returns a template, so checking text length or specific marker
            if "Prediction Results" in response.text or "table" in response.text:
                print_status("Batch Predict", True, "Received HTML response with results")
            else:
                print_status("Batch Predict", True, f"Response received (Length: {len(response.text)})")
        else:
            print_status("Batch Predict", False, f"Status {response.status_code}")
    except Exception as e:
        print_status("Batch Predict", False, str(e))

if __name__ == "__main__":
    print("🚀 Starting Unified Verification...")
    test_health()
    test_training()
    test_single_prediction()
    test_batch_prediction()
    print("\n✨ Verification Complete")
