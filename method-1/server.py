from flask import Flask, request
from flask_cors import CORS
from urllib.parse import unquote
import json
import time

from comments_by_elements import *
from comments_by_context import *


app = Flask(__name__)
CORS(app)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["timeout"] = (10 * 60) # 10 minutes

# tool_for_context = None, None
tool_for_elements = Comments_By_Elements()
tool_for_context = Comments_By_Context()

@app.route("/")
def hello_world():
    res_dic = {"a":"hello worldds"}
    return json.dumps(res_dic)

@app.route("/analyze_image_url", methods=['GET', 'POST'])
def analyze_image_url():
    image_url = unquote(request.json.get("image_url"))

    start_time = time.time()
    (elements, comments_by_elements) = tool_for_elements.analyse_image_url(image_url)
    comment_by_context = tool_for_context.analyse_image_url(image_url)

    end_time = time.time()
    print(f"time: {end_time - start_time}s")

    res_dict = {"code": 1}
    data_dict = {}
    data_dict["elements"] = elements
    data_dict["comments_by_elements"] = comments_by_elements

    data_dict["comment_by_context"] = comment_by_context
    res_dict["data"] = data_dict
    return json.dumps(res_dict)

@app.route("/analyze_image_data", methods=['GET', 'POST'])
def analyze_image_data():
    img_data = request.files['img']
    (elements, comments_by_elements) = tool_for_elements.analyse_image_data(img_data)
    comment_by_context = tool_for_context.analyse_image_data(img_data)

    res_dict = {"code": 1}
    data_dict = {}
    data_dict["elements"] = elements
    data_dict["comments_by_elements"] = comments_by_elements

    data_dict["comment_by_context"] = comment_by_context
    res_dict["data"] = data_dict
    return json.dumps(res_dict)

if __name__ == '__main__':
    app.run()