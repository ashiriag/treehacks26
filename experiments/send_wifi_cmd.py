# import requests
# import time

# ESP_IP = "172.20.10.12"  # replace with your ESP IP

# def set_drawing(state: bool):
#     url = f"http://{ESP_IP}/drawing"
#     payload = {"drawing": state}
#     try:
#         requests.post(url, json=payload, timeout=0.1)
#     except:
#         pass

# # Test
# while True:
#     set_drawing(True)
#     time.sleep(1)
#     set_drawing(False)
#     time.sleep(1)

import requests
import time

ESP_IP = "172.20.10.12"
URL = f"http://{ESP_IP}/drawing"

def set_drawing(state: bool):
    # Do NOT reuse a Session here (ESP8266 often closes keep-alive)
    for attempt in range(5):
        try:
            r = requests.post(URL, json={"drawing": state}, timeout=2)
            # optional: print(r.status_code, r.text)
            return True
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt+1}/5): {e}")
            time.sleep(0.2)
    return False

while True:
    print("Turning on drawing...")
    set_drawing(True)
    time.sleep(2)

    print("Turning off drawing...")
    set_drawing(False)
    time.sleep(2)
