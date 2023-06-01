import replicate
from nltk.stem import WordNetLemmatizer


REPLICATE_API_TOKEN = "The API Token from your Replicate account"

class Material_By_Blip:
    def __init__(self, k) -> None:

        self.k = k
        
        self.MINI_GPT_MODEL = "joehoover/instructblip-vicuna13b:c4c54e3c8c97cd50c2d2fec9be3b6065563ccf7d43787fb99f84151b867178fe"
        self.BLIP2_MODEL = "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608"
        self.replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        self.wnl = WordNetLemmatizer()

    def obtain_materials_from_image_url(self, image_url, count=5):
        return self.__analyse_image(image=image_url, count=count)
    
    def obtain_materials_from_image_data(self, image_data, count=5):
        return self.__analyse_image(image=image_data, count=count)
    
    def __analyse_image(self, image, count=5):        
        context_sentence = self.__get_context_of_image(image)

        elements = self.__get_elements_of_image(image, count)

        print("\n======context_sentence======")
        print(context_sentence)
        print("\n======elements======")
        print(elements)

        return (context_sentence, elements)
    
    def __get_context_of_image(self, image):
        input_params = {
            "prompt": "Describe this picture in detail.",
            "img": image
        }

        return self.__run_replicate(self.MINI_GPT_MODEL, input_params)

    def __get_elements_of_image(self, image, count=5):        
        input_params = {
            "question": f"What are the {count} major elements in the picture?",
            "image": image
        }
        generated_text = self.__run_replicate(self.BLIP2_MODEL, input_params)
        print("__get_elements_of_image:", generated_text)
        
        raw_elements = generated_text.split(",")
        res = []
        for raw_ele in raw_elements:
            raw_ele = raw_ele.strip().lower()
            if (raw_ele.startswith("and ")):
                raw_ele = raw_ele[4:]
            if (raw_ele.startswith("the ")):
                raw_ele = raw_ele[4:]
            if (raw_ele.startswith("a ")):
                raw_ele = raw_ele[2:]
            if (raw_ele == "text"):
                continue
            raw_ele = self.wnl.lemmatize(raw_ele, 'n')
            res.append(raw_ele)
        
        return self.__remove_duplicates_in_list(res)
    
    def __run_replicate(self, model, input_params):
        output = self.replicate_client.run(
            model,
            input=input_params
        )

        result = ""
        for item in output:
            # print(item)
            result += item
        # print("result:", result)
        return result
    
    def __remove_duplicates_in_list(self, ori_lst):
        new_list = list(set(ori_lst))
        new_list.sort(key=ori_lst.index)
        return new_list
    

if __name__ == '__main__':
    blip_tool = Material_By_Blip()
    res = blip_tool.obtain_materials_from_image_url("http://n.sinaimg.cn/sinacn15/250/w640h410/20180318/6d63-fyshfur2581706.jpg")
    # image_path = "/Users/albus/Downloads/AI-works/AI-Test/Z-Images/Cafe.jpg"
    # res = blip_tool.obtain_materials_from_image_data(open(image_path, "rb"))
    print("res:", res)