
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

image_url = "https://images.france.fr/zeaejvyq9bhj/J4oNHeCqEINKfQaiHyD5N/62b2c5ed60090615ff8ef112921d3097/AdobeStock_120637322-crop.jpg?w=1120&h=490&q=70&fl=progressive&fit=fill"
the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

def get_objects_by_blip2(image, k=5):
    prompt = f"Question: What are the {k} major elements in the picture? Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float32)
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("generated_text:", generated_text)
    return generated_text

_ = get_objects_by_blip2(image=the_image)