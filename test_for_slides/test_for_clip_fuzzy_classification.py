import clip
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_attractiveness_of_face(face_image, gender):
    cls_names = ["beautiful", "ugly"]
    if (gender == "man"):
        cls_names = ["handsome", "ugly"]

    image = preprocess(face_image).unsqueeze(0).to(device)
    text = clip.tokenize(cls_names).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("probs:", probs)
        return (probs.argmax(), probs.max())

image_url = "https://i.pinimg.com/564x/fc/31/97/fc319721d6e27a93468e38656a4e3092.jpg"
the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
_ = get_attractiveness_of_face(the_image, "woman")