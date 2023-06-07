from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import requests
import random
import openai
from nltk.corpus import wordnet

class Comments_By_Elements:
    def __init__(self) -> None:
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

        OPENAI_API_KEY = "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"
        openai.api_key = OPENAI_API_KEY
    
    def analyse_image_url(self, image_url):
        the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        return self.__analyse_image(image=the_image)
    
    def analyse_image_data(self, image_data):
        the_image = Image.open(image_data).convert("RGB")
        return self.__analyse_image(image=the_image)
    
    def __analyse_image(self, image):
        element_list = self.__get_elements_by_vit(image)
        purged_element_list = self.__purge_elements(element_list)
        the_adj_words_list = self.__get_adj_words(purged_element_list)
        the_antonym_word_list = self.__get_antonym_word_list(the_adj_words_list)
        the_combined_antonyms_with_items = self.__combined_antonyms_with_items(the_antonym_word_list, purged_element_list)
        the_anology_list = self.__get_analogies_by_openai(the_combined_antonyms_with_items)
        the_seed_sentence_list = self.__create_seed_sentence_list(the_anology_list, the_adj_words_list, purged_element_list)
        the_paraphrased_sentences = self.__paraphrase_sentences(the_seed_sentence_list)
        return (purged_element_list, the_paraphrased_sentences)

    def __get_elements_by_vit(self, image, k=5):
        inputs = self.image_processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze()
        # print("logits:", logits)
        elements_res = []
        # model predicts one of the 1000 ImageNet classes
        vs, ds = torch.topk(logits, k)
        # print("vs:", vs)
        for i in range(ds.shape[0]):
            d = ds[i]
            predicted_label = self.model.config.id2label[d.item()]
            # print("%.2f, %s" % (vs[i].item(), predicted_label))
            elements_res.append(predicted_label)

        return elements_res
    
    def __purge_elements(self, lst):
        res = []
        for i in range(len(lst)):
            ele = lst[i]
            seperator = ","
            if (seperator in ele):
                parts = ele.split(seperator)
                rdm_part = random.choice(parts).strip()
                res.append(rdm_part)
            else:
                res.append(ele)
        return res
    
    def __generate_adj_words_str(self, lst):
        res = ""
        for i in range(len(lst)):
            ele = lst[i]
            res += ('\"' + ele + '\"')
            if ((i + 1) != len(lst)):
                res += ", "
        return f"[{res}]"

    def __generate_adj_words_prompt(self, lst):
        res = f"List 5 non-negative adjective words for each of these noun words respectively: {lst}. Give result in python 2-d array form, e.g., [[\"adj1\", \"adj2\", ..., \"adj5\"], [\"adj1\", \"adj2\", ..., \"adj5\"], [\"adj1\", \"adj2\", ..., \"adj5\"]]."
        return res

    def __get_adj_words(self, lst):
        prompt = self.__generate_adj_words_prompt(lst)
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        adj_words_list_str = answer_item["text"].strip()
        # print("adj_words_list_str:", adj_words_list_str)
        adj_words_list = eval(adj_words_list_str)
        return adj_words_list
    
    # Get antonym
    def __get_antonym_of_single_adj_word(self, word_item):
        res = None
        if (len(wordnet._morphy(word_item, pos="a")) == 0):
            return res
        
        lemma = wordnet.lemma(word_item + ".a.01." + word_item)
        # print("lemma:", lemma)
        if (lemma is None):
            return res
        synset = lemma.synset()
        if (synset.pos() == "a"):
            antonyms = lemma.antonyms()
            # print("antonyms:", antonyms)
            if (len(antonyms) > 0):
                antom = antonyms[0]
                # print("antom:", antom.name())
                res = antom.name()
        else:
            main_synset = None
            for similar_synset in synset.similar_tos():
                if similar_synset.pos() == 'a':
                    main_synset = similar_synset
                    break

            antonyms = []
            if main_synset:
                for lemma in main_synset.lemmas():
                    for antonym in lemma.antonyms():
                        antonyms.append(antonym.name())

            # print("antonyms:", antonyms)
            if (len(antonyms) > 0):
                res = antonyms[0]
        
        return res
    
    def __get_antonym_by_openai(self, word_item):
        # prompt = f"Find an antonym for the adjective word \"{word_item}\""
        prompt = f"Find an antonym for the adjective word \"{word_item}\". Give result without \".\""
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        antonym_word = answer_item["text"].strip().lower()
        if (antonym_word.endswith(".")):
            antonym_word = antonym_word[:-1]
        return antonym_word
    
    def __get_antonym_word_list(self, lst):
        res = []
        for sub_lst in lst:
            sub_res = []
            for adj_word in sub_lst:
                antonym_word = self.__get_antonym_of_single_adj_word(adj_word)
                if (antonym_word is None):
                    antonym_word = self.__get_antonym_by_openai(adj_word)
                sub_res.append(antonym_word)
            res.append(sub_res)
        return res
    
    # Get anologies from the combination of negative adjective word & the item
    def __combined_antonyms_with_items(self, antonyms, items):
        res = []
        for i in range(len(items)):
            sub_res = []
            sub_antonyms = antonyms[i]
            item = items[i]
            for antonym in sub_antonyms:
                term = (antonym + " " + item)
                sub_res.append(term)
            res.append(sub_res)
        return res
    
    def __get_analogies_by_openai(self, descriptive_terms):
        prompt = f"Like that \"A snail in a swimming pool\" is an analogy to \"slow boat\", what are the things that can be used as anologies to {descriptive_terms}?  Give result in python 2-d array form, e.g., [[\"ans1\", \"ans2\", \"ans3\", \"ans4\", \"ans5\"], [\"ans1\", \"ans2\", \"ans3\", \"ans4\", \"ans5\"], ...]."

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("\nresponse:", response)
        choices = response["choices"]
        answer_item = choices[0]
        anology_list_str = answer_item["text"].strip()
        # print("\nanology_list_str:", anology_list_str)
        anology_list = eval(anology_list_str)
        # print("\nanology_list:", anology_list)
        return anology_list
    
    # Paraphrasing
    # seed_sentence_list
    def __create_seed_sentence_list(self, anology_list, adj_words_list, items):
        res = []
        for i in range(len(items)):
            sub_res = []
            item = items[i]
            sub_adj_words = adj_words_list[i]
            sub_analogies = anology_list[i]        
            for j in range(len(sub_analogies)):
                analogy = sub_analogies[j]
                adj_word = sub_adj_words[j]            
                seed_sentence = f"such a {adj_word} {item}, like {analogy.lower()}"
                sub_res.append(seed_sentence)
            res.append(sub_res)
        return res
    
    def __paraphrase_seed_sentence(self, seed_sentence):
        prompt = f"paraphrase the sentence: \"{seed_sentence}\", in another form, without transition words like \"but\", \"yet\", etc.",

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        answer_str = answer_item["text"].strip()
        
        return answer_str
    
    def __paraphrase_sentences(self, seed_sentence_list):
        res = []
        for i in range(len(seed_sentence_list)):
            sub_res = []
            sub_seed_sentence_list = seed_sentence_list[i]
            for seed_sentence in sub_seed_sentence_list:
                paraphrased_sentence = self.__paraphrase_seed_sentence(seed_sentence)
                sub_res.append(paraphrased_sentence)
            res.append(sub_res)
        return res
    

if __name__ == '__main__':
    tool_for_elements = Comments_By_Elements()
    res = tool_for_elements.analyse_image_url("http://n.sinaimg.cn/sinacn15/250/w640h410/20180318/6d63-fyshfur2581706.jpg")
    print("res:", res)