from attr import dataclass
from flask import request, Flask, jsonify, make_response,render_template,send_from_directory
from flask_cors import CORS, cross_origin
import os
import gc
import sys
import json
import time
import pickle
import shutil
import base64
import numpy as np
from utils import *

sys.path.append('..')
sys.path.append('.')

# flask for API server
app = Flask(__name__,static_folder='../Frontend')
cors = CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

# TODO:from where to get config path?
config_file ='/path/to/config.json'

@dataclass
class TempConfig:
    TASK_TYPE: str | None = None
    CONTENT_PATH: str | None = None
    EPOCH_START: int = 0
    EPOCH_END: int = 30
    EPOCH_PERIOD: int = 1

@app.route("/", methods=["GET", "POST"])
def GUI():
    return send_from_directory(app.static_folder, 'index.html')

# Func: get iteration structure
@app.route('/get_itertaion_structure', methods=["POST", "GET"])
@cross_origin()
def get_tree():
    config = TempConfig()
    json_data = []
    previous_epoch = ""
    for epoch in range(config.EPOCH_START, config.EPOCH_END + 1, config.EPOCH_PERIOD):
        json_data.append({
            "value": epoch,
            "name": 'Epoch',
            "pid": previous_epoch if previous_epoch else ""
        })
        previous_epoch = epoch
    return make_response(jsonify({"structure":json_data}), 200)

# Func: load projection result of one epoch
@app.route('/updateProjection', methods = ["POST", "GET"])
@cross_origin()
def update_projection():
    # search filter
    req = request.get_json()
    iteration = int(req['iteration'])
    predicates = req['predicates']
    indicates = list(range(100)) # we now don't use req['selectedPoints'] to filter in backend

    # load config from config_file    
    config = TempConfig(req['taskType'], req['contentPath'])
    
    # load visualization result of one epoch
    if config.TASK_TYPE == 'classification' or config.TASK_TYPE == 'non-classification':

        embedding_2d, grid, decision_view, label_name_dict, label_color_list, label_list, max_iter, training_data_index, \
        testing_data_index, eval_new, prediction_list, selected_points, error_message_projection, color_list, \
            confidence_list = update_epoch_projection(config, iteration, predicates, indicates)
        
        # make response and return
        grid = np.array(grid)
        color_list = color_list.tolist()
        return make_response(jsonify({'result': embedding_2d, 
                                    'grid_index': grid.tolist(), 
                                    'grid_color': 'data:image/png;base64,' + decision_view,
                                    'label_name_dict':label_name_dict,
                                    'label_color_list': label_color_list, 
                                    'label_list': label_list,
                                    'maximum_iteration': max_iter, 
                                    'training_data': training_data_index,
                                    'testing_data': testing_data_index, 
                                    'evaluation': eval_new,
                                    'prediction_list': prediction_list,
                                    "selectedPoints":selected_points.tolist(),
                                    "errorMessage": error_message_projection,
                                    "color_list": color_list,
                                    "confidence_list": confidence_list
                                    }), 200)
    elif config.TASK_TYPE == 'Umap-Neighborhood':
        result = get_umap_neighborhood_epoch_projection(config.CONTENT_PATH, iteration, predicates, indicates)
        return make_response(jsonify(result), 200)
    else:
        return make_response(jsonify({'error': 'TaskType not found'}), 400)


# Func: get sprite or text of one sample
@app.route('/spriteImage', methods = ["GET"])
@cross_origin()
def sprite_image():
    index = int(request.args.get("index"))
    
    # load config from config_file
    config = initailize_config(config_file)
    if config.DATA_TYPE == "image":
        pic_save_dir_path = os.path.join(config.CONTENT_PATH, "Dataset","sprites", "{}.png".format(index))
        img_stream = ""
        with open(pic_save_dir_path, 'rb') as img_f:
            img_stream = img_f.read()
            img_stream = base64.b64encode(img_stream).decode()
        return make_response(jsonify({"imgUrl":'data:image/png;base64,' + img_stream}), 200)
    elif config.DATA_TYPE == "text":
        if config.SHOW_LABEL:
            if index % 2 == 0: # source
                text_save_dir_path = os.path.join(config.CONTENT_PATH, "Dataset","source", "{}.txt".format(int(index/2)))
            else: # target
                text_save_dir_path = os.path.join(config.CONTENT_PATH, "Dataset","target", "{}.txt".format(int(index/2)))
        else:
            text_save_dir_path = os.path.join(config.CONTENT_PATH, "Dataset","source", "{}.txt".format(index))
        
        sprite_texts = ''
        with open(text_save_dir_path, 'r') as text_f:
            sprite_texts = text_f.read()
        return make_response(jsonify({"texts": sprite_texts}), 200)    
    else:
        raise ValueError("Invalid data type in config")

@app.route('/spriteText', methods = ["GET"])
@cross_origin()
def sprite_text():
    index = int(request.args.get("index"))
    
    # load config from config_file
    config = initailize_config(config_file)
    
    if config.SHOW_LABEL:
        if index % 2 == 0: # source
            text_save_dir_path = os.path.join(config.CONTENT_PATH, "Dataset","source", "{}.txt".format(int(index/2)))
        else: # target
            text_save_dir_path = os.path.join(config.CONTENT_PATH, "Dataset","target", "{}.txt".format(int(index/2)))
    else:
        text_save_dir_path = os.path.join(config.CONTENT_PATH, "Dataset","source", "{}.txt".format(index))
    
    sprite_texts = ''
    if os.path.exists(text_save_dir_path):
        with open(text_save_dir_path, 'r') as text_f:
            sprite_texts = text_f.read()
    else:
        print("File does not exist:", text_save_dir_path)
  
    response_data = {
        "texts": sprite_texts
    }
    return make_response(jsonify(response_data), 200)



    
def check_port_inuse(port, host):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, port))
        return True
    except socket.error:
        return False
    finally:
        if s:
            s.close()
# for contrast
if __name__ == "__main__":
    import socket
    # hostname = socket.gethostname()
    # ip_address = socket.gethostbyname(hostname)
    ip_address = '0.0.0.0'
    port = 5000
    while check_port_inuse(port, ip_address):
        port = port + 1

    app.run(host=ip_address, port=port)