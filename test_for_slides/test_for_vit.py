
from transformers import AutoImageProcessor, ViTForImageClassification
import torch
import requests
from PIL import Image

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

image_url = "https://images.france.fr/zeaejvyq9bhj/J4oNHeCqEINKfQaiHyD5N/62b2c5ed60090615ff8ef112921d3097/AdobeStock_120637322-crop.jpg?w=1120&h=490&q=70&fl=progressive&fit=fill"
the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

def get_objects_by_vit(image, k=5):
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()        
        elements_res = []
        vs, ds = torch.topk(logits, k)        
        for i in range(ds.shape[0]):
            d = ds[i]
            predicted_label = model.config.id2label[d.item()]
            elements_res.append(predicted_label)
            print("%.2f, %s" % (vs[i].item(), predicted_label))
        
        return elements_res
    
_ = get_objects_by_vit(image=the_image)