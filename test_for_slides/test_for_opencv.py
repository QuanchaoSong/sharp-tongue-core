

import cv2
from PIL import Image, ImageDraw
import numpy as np
import requests

frontalface_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(frontalface_path)

image_url = "https://i.pinimg.com/564x/fc/31/97/fc319721d6e27a93468e38656a4e3092.jpg"
the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

draw = ImageDraw.Draw(the_image)
faces = face_cascade.detectMultiScale(np.array(the_image), scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))

for (x,y,w,h) in faces:
    draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 5), width=2)

the_image.show()