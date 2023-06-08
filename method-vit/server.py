from flask import Flask, request
from flask_cors import CORS
from urllib.parse import unquote
import json
import time
from io import BufferedReader

from comments_by_elements import *
from comments_by_context import *
from comments_for_face import *
from comments_by_context_replicate import *


app = Flask(__name__)
CORS(app)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["timeout"] = (10 * 60) # 10 minutes

# tool_for_context = None, None
tool_for_elements = Comments_By_Elements()
# tool_for_context = Comments_By_Context()
tool_for_context = Comments_By_Context_Replicate()
tool_for_face = Comments_For_Face()

@app.route("/")
def hello_world():
    res_dic = {"a":"hello worldds"}
    return json.dumps(res_dic)

@app.route("/analyze_image_url", methods=['GET', 'POST'])
def analyze_image_url():
    image_url = unquote(request.json.get("image_url"))

    start_time = time.time()
    (elements, comments_by_elements) = tool_for_elements.analyse_image_url(image_url)
    comments_by_context = tool_for_context.analyse_image_url(image_url)
    (face_exist, face_desc, comments_for_face) = tool_for_face.analyse_image_url(image_url)

    end_time = time.time()
    print(f"time: {end_time - start_time}s")

    res_dict = {"code": 1}
    data_dict = {}
    data_dict["elements"] = elements
    data_dict["comments_by_elements"] = comments_by_elements

    data_dict["comments_by_context"] = comments_by_context
    face_dict = {}
    if (face_exist):
        face_dict["face_exist"] = 1
        face_dict["face_desc"] = face_desc
        face_dict["comments_for_face"] = comments_for_face
    else:
        face_dict["face_exist"] = 0
    data_dict["face"] = face_dict
    res_dict["data"] = data_dict

    return json.dumps(res_dict)

@app.route("/analyze_image_data", methods=['GET', 'POST'])
def analyze_image_data():
    img_data = request.files['img']
    img_data.name = img_data.filename
    img_data = BufferedReader(img_data)  
    (elements, comments_by_elements) = tool_for_elements.analyse_image_data(img_data)
    comments_by_context = tool_for_context.analyse_image_data(img_data)
    (face_exist, face_desc, comments_for_face) = tool_for_face.analyse_image_data(img_data)

    res_dict = {"code": 1}
    data_dict = {}
    data_dict["elements"] = elements
    data_dict["comments_by_elements"] = comments_by_elements

    data_dict["comments_by_context"] = comments_by_context
    face_dict = {}
    if (face_exist):
        face_dict["face_exist"] = 1
        face_dict["face_desc"] = face_desc
        face_dict["comments_for_face"] = comments_for_face
    else:
        face_dict["face_exist"] = 0
    data_dict["face"] = face_dict
    res_dict["data"] = data_dict
    
    return json.dumps(res_dict)

if __name__ == '__main__':
    app.run()