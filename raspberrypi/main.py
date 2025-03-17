import camera_car_detect
from camera_car_detect import model_initialize, analysis, show

from picamera2 import Picamera2

import requests


# MODEL = "yolov8x.pt"
# MODEL = "yolov8x_320.pt"
# MODEL = "yolov8n_320.pt"
MODEL = "yolov8n.pt"

camera = Picamera2()
# camera.configure(picam2.create_still_configuration(main={"size": picture_size}, lores={"size": picture_size}))
camera.start()

model = model_initialize(MODEL)

RASPI_ID = '0'

while True:
    camera.capture_file("temp.PNG")
    
    detections, labels = analysis(model)
    print(labels)
    count = 0
    for label in labels:
        if 'cell' in label or 'remote' in label or 'car' in label:
            count += 1
    print(count)
    show(detections, labels)
    
    url = 'http://192.168.0.5:5000/traffic_input'
    data = {
        'pi_id': RASPI_ID,
        'traffic': count
    }

    response = requests.post(url, json=data)
    # print("received: ", response.json())
