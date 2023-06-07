
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32
        )
model.to(device)

image_url = "https://4.bp.blogspot.com/-TpyTJovVB-k/UjI1fPzWgqI/AAAAAAAAACs/vk5gDxo9vOc/s1600/Pigs+eating+compost+(8).JPG"
the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

def get_objects_by_blip2(image):
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float32)
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("generated_text:", generated_text)
    return generated_text

_ = get_objects_by_blip2(image=the_image)