import requests


url = 'http://192.168.0.5:5000/traffic_input'
data = {
    'pi_id': '0',
    'traffic': '63'
}

response = requests.post(url, json=data)
print("received: ", response.json())
