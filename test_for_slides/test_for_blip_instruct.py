import replicate
import torch


REPLICATE_API_TOKEN = "Your Replicate API token"
MINI_GPT_MODEL = "joehoover/instructblip-vicuna13b:c4c54e3c8c97cd50c2d2fec9be3b6065563ccf7d43787fb99f84151b867178fe"

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

def get_context_of_image(image):
    input_params = {
        "prompt": "Describe this picture in detail.",
        "img": image
    }

    return run_replicate(MINI_GPT_MODEL, input_params)

def run_replicate(model, input_params):
    output = replicate_client.run(
        model,
        input=input_params
    )

    result = ""
    for item in output:
        # print(item)
        result += item
    # print("result:", result)
    return result

image_url = "https://4.bp.blogspot.com/-TpyTJovVB-k/UjI1fPzWgqI/AAAAAAAAACs/vk5gDxo9vOc/s1600/Pigs+eating+compost+(8).JPG"
res = get_context_of_image(image_url)
print("context in detail:", res)