

from deepface import DeepFace
import numpy as np
from PIL import Image
import requests


image_url = "https://i.pinimg.com/564x/fc/31/97/fc319721d6e27a93468e38656a4e3092.jpg"
the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

face_analysis = DeepFace.analyze(img_path=np.array(the_image))
print("face_analysis:", face_analysis)