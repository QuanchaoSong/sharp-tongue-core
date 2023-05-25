from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

class Material_By_Blip:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32
        )
        self.model.to(self.device)

    def obtain_materials_from_image_url(self, image_url):
        the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        return self.__analyse_image(image=the_image)
    
    def obtain_materials_from_image_data(self, image_data):
        the_image = Image.open(image_data).convert("RGB")
        return self.__analyse_image(image=the_image)
    
    def __analyse_image(self, image):        
        context_sentence = self.__get_context_of_image(image)

        elements = self.__get_elements_of_image(image)

        return (context_sentence, elements)
    
    def __get_context_of_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float32)
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    def __get_elements_of_image(self, image):        
        prompt = "Question: What are the 5 major elements in the picture? Answer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float32)
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print("generated_text:", generated_text)

        raw_elements = generated_text.split(",")
        res = []
        for raw_ele in raw_elements:
            raw_ele = raw_ele.strip().lower()
            if (raw_ele.startswith("and ")):
                raw_ele = raw_ele[4:]
            if (raw_ele.startswith("the ")):
                raw_ele = raw_ele[4:]
            if (raw_ele != "text"):
                res.append(raw_ele)
        return res    
    

if __name__ == '__main__':
    blip_tool = Material_By_Blip()
    res = blip_tool.obtain_materials_from_image_url("http://n.sinaimg.cn/sinacn15/250/w640h410/20180318/6d63-fyshfur2581706.jpg")
    print("res:", res)