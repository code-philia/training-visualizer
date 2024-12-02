import os
import json
from secrets import token_urlsafe
import time
import csv
import numpy as np
import sys
import pickle
import base64

vis_path = ".."
sys.path.append(vis_path)
from context import VisContext, ActiveLearningContext, AnormalyContext
from strategy import DeepDebugger, TimeVis, tfDeepVisualInsight, DVIAL, tfDVIDenseAL, TimeVisDenseAL, Trustvis, DeepVisualInsight
from singleVis.eval.evaluate import rank_similarities_and_color, evaluate_isAlign, evaluate_isNearestNeighbour, evaluate_isAlign_single, evaluate_isNearestNeighbour_single
from sklearn.cluster import KMeans
from scipy.special import softmax
import matplotlib.pyplot as plt
import time
import torch
"""Interface align"""

def initialize_strategy(CONTENT_PATH, VIS_METHOD, SETTING, dense=False):
    # initailize strategy (visualization method)
    with open(os.path.join(CONTENT_PATH, "config.json"), "r") as f:
        conf = json.load(f)
    # VIS_METHOD = "DVI" 
    config = conf["DVI"]
    error_message = ""
    strategy = Trustvis(CONTENT_PATH, config)
    try:
        if SETTING == "normal" or SETTING == "abnormal":
            if VIS_METHOD == "Trustvis":
                strategy = Trustvis(CONTENT_PATH, config)
            elif VIS_METHOD == "DVI":
                strategy = tfDeepVisualInsight(CONTENT_PATH, config)
            elif VIS_METHOD == "TimeVis":
                strategy = TimeVis(CONTENT_PATH, config)
            elif VIS_METHOD == "DeepDebugger":
                strategy = DeepDebugger(CONTENT_PATH, config)
            else:
                error_message += "Unsupported visualization method\n"
        elif SETTING == "active learning":
            if dense:
                if VIS_METHOD == "DVI":
                    strategy = tfDVIDenseAL(CONTENT_PATH, config)
                elif VIS_METHOD == "TimeVis":
                    strategy = TimeVisDenseAL(CONTENT_PATH, config)
                else:
                    error_message += "Unsupported visualization method\n"
            else:
                strategy = DVIAL(CONTENT_PATH, config)
        
        else:
            error_message += "Unsupported setting\n"
    except Exception as e:
        error_message += "mismatch in input vis method and current visualization model\n"
    return strategy, error_message

def initialize_context(strategy, setting):
    if setting == "normal":
        context = VisContext(strategy)
    elif setting == "active learning":
        context = ActiveLearningContext(strategy)
    elif setting == "abnormal":
        context = AnormalyContext(strategy)
    else:
        raise NotImplementedError
    return context

def initialize_backend(CONTENT_PATH, VIS_METHOD, SETTING, dense=False):
    """ initialize backend for visualization

    Args:
        CONTENT_PATH (str): the directory to training process
        VIS_METHOD (str): visualization strategy
            "DVI", "TimeVis", "DeepDebugger",...
        setting (str): context
            "normal", "active learning", "dense al", "abnormal"

    Raises:
        NotImplementedError: _description_

    Returns:
        backend: a context with a specific strategy
    """
    strategy, error_message = initialize_strategy(CONTENT_PATH, VIS_METHOD, SETTING, dense)
    print("contenePath", CONTENT_PATH)
    context = initialize_context(strategy=strategy, setting=SETTING)
    return context, error_message



def check_labels_match_alldata(labels, all_data, error_message):
    if (len(labels) != len(all_data)):
        error_message += "The total number of data labels doesn't match with the total number of data samples!\n"
    return error_message

def get_embedding(context, all_data, EPOCH):
    embedding_path = get_embedding_path(context, EPOCH)

    if os.path.exists(embedding_path):
        embedding_2d = np.load(embedding_path, allow_pickle=True) 
    else:
        embedding_2d = context.strategy.projector.batch_project(EPOCH, all_data)
        np.save(embedding_path, embedding_2d)
    return embedding_2d


def get_custom_embedding(context, all_data, EPOCH):

    embedding_2d = context.strategy.projector.batch_project(EPOCH, all_data)

    return embedding_2d

def get_embedding_path(context, EPOCH):
    embedding_path = os.path.join(context.strategy.data_provider.checkpoint_path(EPOCH), "embedding.npy")
    return embedding_path

def check_embedding_match_alldata(embedding_2d, all_data, error_message):
    if (len(embedding_2d) != len(all_data)):
        error_message += "The total number of projected points doesn't match with the total number of data samples!\n"
    return error_message

def check_config_match_embedding(training_data_number, testing_data_number, embedding_2d, error_message):
    if ((training_data_number + testing_data_number) != len(embedding_2d)):
        error_message += "config file's setting of total number of data samples and total number of projected points don't match!\n"
    return error_message

def get_grid_bfig(context, EPOCH, embedding_2d):
    bgimg_path = os.path.join(context.strategy.data_provider.checkpoint_path(EPOCH), "bgimg.png")
    scale_path = os.path.join(context.strategy.data_provider.checkpoint_path(EPOCH), "scale.npy")
    # grid_path = os.path.join(context.strategy.data_provider.checkpoint_path(EPOCH), "grid.pkl")
    if os.path.exists(bgimg_path) and os.path.exists(scale_path):
        # with open(os.path.join(grid_path), "rb") as f:
        #     grid = pickle.load(f)
        with open(bgimg_path, 'rb') as img_f:
            img_stream = img_f.read()
        b_fig = base64.b64encode(img_stream).decode()
        grid = np.load(scale_path)
    else:
        x_min, y_min, x_max, y_max, b_fig = context.strategy.vis.get_background(EPOCH, context.strategy.config["VISUALIZATION"]["RESOLUTION"])
        grid = [x_min, y_min, x_max, y_max]
        # formating
        grid = [float(i) for i in grid]
        b_fig = str(b_fig, encoding='utf-8')
        # save results, grid and decision_view
        # with open(grid_path, "wb") as f:
        #     pickle.dump(grid, f)
        np.save(get_embedding_path(context, EPOCH), embedding_2d)
    return grid, b_fig

def get_eval_new(context, EPOCH):
    eval_new = dict()
    file_name = context.strategy.config["VISUALIZATION"]["EVALUATION_NAME"]
    save_eval_dir = os.path.join(context.strategy.data_provider.model_path, file_name + ".json")
    if os.path.exists(save_eval_dir):
        evaluation = context.strategy.evaluator.get_eval(file_name=file_name)
        eval_new["train_acc"] = evaluation["train_acc"][str(EPOCH)]
        eval_new["test_acc"] = evaluation["test_acc"][str(EPOCH)]
    else:
        eval_new["train_acc"] = 0
        eval_new["test_acc"] = 0
    return eval_new

def get_train_test_data(context, EPOCH):
    train_data = context.train_representation_data(EPOCH)
    test_data = context.test_representation_data(EPOCH)
    if train_data is None:
        all_data = test_data
    elif test_data is None:
        all_data = train_data
    else:
        all_data = np.concatenate((train_data, test_data), axis=0)
    # print(len(test_data))
    # print(len(train_data))
    return all_data

def get_train_test_label(context, EPOCH):
    train_labels = context.train_labels(EPOCH)
    test_labels = context.test_labels(EPOCH)
    train_data = context.train_representation_data(EPOCH) 
    test_data = context.test_representation_data(EPOCH)
    if train_labels is None:
        train_labels = np.zeros(len(train_data) if train_data is not None else 0, dtype=int)
    if test_labels is None:
        test_labels = np.zeros(len(test_data) if test_data is not None else 0, dtype=int)
    print("errorlabels", train_labels, test_labels)
    labels = np.concatenate((train_labels, test_labels), axis=0).astype(int)
        
    return labels

def get_selected_points(context, predicates, EPOCH, training_data_number, testing_data_number):
    selected_points = np.arange(training_data_number + testing_data_number)
    for key in predicates.keys():
        if key == "label":
            tmp = np.array(context.filter_label(predicates[key]))
        elif key == "type":
            tmp = np.array(context.filter_type(predicates[key], int(EPOCH)))
        else:
            tmp = np.arange(training_data_number + testing_data_number)    
        selected_points = np.intersect1d(selected_points, tmp)

    return selected_points

def get_properties(context, training_data_number, testing_data_number, training_data_index, EPOCH):
    properties = np.concatenate((np.zeros(training_data_number, dtype=np.int16), 2*np.ones(testing_data_number, dtype=np.int16)), axis=0)
    lb = context.get_epoch_index(EPOCH)
    ulb = np.setdiff1d(training_data_index, lb)
    properties[ulb] = 1
    return properties

def get_coloring(context, EPOCH, ColorType):
    label_color_list = []
    train_data =  context.train_representation_data(EPOCH) 
    test_data =  context.test_representation_data(EPOCH) 
    labels = get_train_test_label(context, EPOCH)
    # coloring method
    if ColorType == "noColoring":
        color = context.strategy.vis.get_standard_classes_color() * 255

        color = color.astype(int)      
        label_color_list = color[labels].tolist()
        # print("alldata", all_data.shape, EPOCH)
        color_list = color
       
    elif ColorType == "singleColoring":
        n_clusters = 20
        save_test_label_dir = os.path.join(context.strategy.data_provider.content_path, 'Testing_data',ColorType+ "label" + str(EPOCH)+".pth")
        save_train_label_dir = os.path.join(context.strategy.data_provider.content_path, 'Training_data',ColorType+ "label" + str(EPOCH)+".pth")
        if os.path.exists(save_train_label_dir):
            labels_kmeans_train = torch.load(save_train_label_dir)
            labels_kmeans_test = torch.load(save_test_label_dir)
        else:
       
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_data)

            labels_kmeans_train = kmeans.labels_
            labels_kmeans_test = kmeans.predict(test_data)
          
            torch.save(torch.tensor(labels_kmeans_train), save_train_label_dir)
            torch.save(torch.tensor(labels_kmeans_test), save_test_label_dir)

        colormap = plt.cm.get_cmap('tab10', n_clusters)
    
        colors_rgb = (colormap(np.arange(n_clusters))[:, :3] * 255).astype(int)  
        label_color_list_train = [colors_rgb[label].tolist() for label in labels_kmeans_train]
        label_color_list_test = [colors_rgb[label].tolist() for label in labels_kmeans_test]
        
        label_color_list = np.concatenate((label_color_list_train, label_color_list_test), axis=0).tolist()
        color_list = colors_rgb

    else:
        return         

    return label_color_list, color_list

def get_comparison_coloring(context_left, context_right, EPOCH_LEFT, EPOCH_RIGHT):
    train_data_left = context_left.train_representation_data(EPOCH_LEFT)
    test_data_left = context_left.test_representation_data(EPOCH_LEFT)
    train_data_right = context_right.train_representation_data(EPOCH_RIGHT)
    test_data_right = context_right.test_representation_data(EPOCH_RIGHT)
    content_path_left = context_left.strategy.data_provider.content_path
    content_path_right = context_right.strategy.data_provider.content_path

    # color depends on the left content path, corresponding right content paths share the same colors with that left content path
    save_test_label_dir_left = os.path.join(content_path_left, 'Testing_data',"doubleColoring" + "label" + str(EPOCH_LEFT) + "_"+ str(EPOCH_RIGHT) + content_path_right[1:].replace(os.sep, '*') + "curr_left"+  ".pth")
    save_train_label_dir_left = os.path.join(content_path_left, 'Training_data',"doubleColoring" + "label" + str(EPOCH_LEFT) + "_"+ str(EPOCH_RIGHT) + content_path_right[1:].replace(os.sep, '*') + "curr_left"+ ".pth")
    save_test_label_dir_right = os.path.join(content_path_right, 'Testing_data',"doubleColoring" + "label" + str(EPOCH_RIGHT) + "_" + str(EPOCH_LEFT) + content_path_left[1:].replace(os.sep, '*') +  "curr_right" + ".pth")
    save_train_label_dir_right = os.path.join(content_path_right, 'Training_data',"doubleColoring" + "label" + str(EPOCH_RIGHT) + "_" + str(EPOCH_LEFT) + content_path_left[1:].replace(os.sep, '*') + "curr_right" + ".pth")

    if os.path.exists(save_test_label_dir_left):
        labels_train_left = torch.load(save_train_label_dir_left)
        labels_test_left = torch.load(save_test_label_dir_left)

        colors_train = rank_similarities_and_color(train_data_left, train_data_right, labels=labels_train_left)
        colors_test = rank_similarities_and_color(test_data_left, test_data_right, labels=labels_test_left)
        print("label_train_;left",labels_train_left)
        print("labels_test_kleft", labels_test_left)
        print("color_train", colors_train)
        print("color_test", colors_test)
       

    else:
        colors_train, labels_train_left = rank_similarities_and_color(train_data_left, train_data_right)
        colors_test, labels_test_left = rank_similarities_and_color(test_data_left, test_data_right)

        torch.save(torch.tensor(labels_train_left), save_train_label_dir_left)
        torch.save(torch.tensor(labels_test_left), save_test_label_dir_left)
        # store the same labels for right content path
        torch.save(torch.tensor(labels_train_left), save_train_label_dir_right)
        torch.save(torch.tensor(labels_test_left), save_test_label_dir_right)

    label_color_list = np.concatenate((colors_train, colors_test), axis=0).tolist()

    return label_color_list    

def update_epoch_projection(context, EPOCH, predicates, TaskType, indicates):
    # TODO consider active learning setting
    error_message = ""
    start = time.time()
    all_data = get_train_test_data(context, EPOCH)
    
    labels = get_train_test_label(context, EPOCH)
    if len(indicates):
        all_data = all_data[indicates]
        labels = labels[indicates]
        
    
    print('labels',labels)
    error_message = check_labels_match_alldata(labels, all_data, error_message)
    
    embedding_2d = get_embedding(context, all_data, EPOCH)
    if len(indicates):
        indicates = [i for i in indicates if i < len(embedding_2d)]
        embedding_2d = embedding_2d[indicates]
    print('all_data',all_data.shape,'embedding_2d',embedding_2d.shape)
    print('indicates', indicates)
    error_message = check_embedding_match_alldata(embedding_2d, all_data, error_message)
    
    training_data_number = context.strategy.config["TRAINING"]["train_num"]
    testing_data_number = context.strategy.config["TRAINING"]["test_num"]
    training_data_index = list(range(training_data_number))
    testing_data_index = list(range(training_data_number, training_data_number + testing_data_number))
    error_message = check_config_match_embedding(training_data_number, testing_data_number, embedding_2d, error_message)
    end = time.time()
    print("beforeduataion", end- start)
    # return the image of background
    # read cache if exists
   
    grid, b_fig = get_grid_bfig(context, EPOCH,embedding_2d)
    # TODO fix its structure
    eval_new = get_eval_new(context, EPOCH)
    start2 = time.time()
    print("midquestion1", start2-end)
    # coloring method    
    label_color_list, color_list = get_coloring(context, EPOCH, "noColoring")
    if len(indicates):
        label_color_list = [label_color_list[i] for i in indicates]

    start1 =time.time()
    print("midquestion2",start1-start2)
    CLASSES = np.array(context.strategy.config["CLASSES"])
    label_list = CLASSES[labels].tolist()
    label_name_dict = dict(enumerate(CLASSES))

    prediction_list = []
    confidence_list = []
    # print("all_data",all_data.shape)
    all_data = all_data.reshape(all_data.shape[0],all_data.shape[1])
    if (TaskType == 'Classification'):
        # check if there is stored prediction and load
        prediction_path = os.path.join(context.strategy.data_provider.checkpoint_path(EPOCH), "modified_ranks.json")
        if os.path.isfile(prediction_path):
            with open(prediction_path, "r") as f:
                predictions = json.load(f)

            for prediction in predictions:
                prediction_list.append(prediction)
        else:
            prediction_origin = context.strategy.data_provider.get_pred(EPOCH, all_data)
            prediction = prediction_origin.argmax(1)

            for i in range(len(prediction)):
                prediction_list.append(CLASSES[prediction[i]])
                top_three_indices = np.argsort(prediction_origin[i])[-3:][::-1]
                conf_list = [(label_name_dict[top_three_indices[j]], round(float(prediction_origin[i][top_three_indices[j]]), 1)) for j in range(len(top_three_indices))]
                confidence_list.append(conf_list)
    else:
        for i in range(len(all_data)):
            prediction_list.append(0)
    
    EPOCH_START = context.strategy.config["EPOCH_START"]
    EPOCH_PERIOD = context.strategy.config["EPOCH_PERIOD"]
    EPOCH_END = context.strategy.config["EPOCH_END"]
    max_iter = (EPOCH_END - EPOCH_START) // EPOCH_PERIOD + 1
    # max_iter = context.get_max_iter()
    
    # current_index = timevis.get_epoch_index(EPOCH)
    # selected_points = np.arange(training_data_number + testing_data_number)[current_index]
    selected_points = get_selected_points(context, predicates, EPOCH, training_data_number, testing_data_number)
    
    properties = get_properties(context, training_data_number, testing_data_number, training_data_index, EPOCH)
    # highlightedPointIndices = []
    #todo highlighpoint only when called with showVis
    # if (TaskType == 'Classification'):
    #     high_pred = context.strategy.data_provider.get_pred(EPOCH, all_data).argmax(1)
    #     inv_high_dim_data = context.strategy.projector.batch_inverse(EPOCH, embedding_2d)
    #     inv_high_pred = context.strategy.data_provider.get_pred(EPOCH, inv_high_dim_data).argmax(1)
    #     highlightedPointIndices = np.where(high_pred != inv_high_pred)[0]
    #     print()
    # else:
        
    #     inv_high_dim_data = context.strategy.projector.batch_inverse(EPOCH, embedding_2d)
    #     # todo, change train data to all data
    #     squared_distances = np.sum((all_data - inv_high_dim_data) ** 2, axis=1)
    #     squared_threshold = 1 ** 2
    #     highlightedPointIndices = np.where(squared_distances > squared_threshold)[0]
    #     print()

    end1 = time.time()
    print("midduration", start1-end)
    print("endduration", end1-start1)
    print("EMBEDDINGLEN", len(embedding_2d))
    return embedding_2d.tolist(), grid, b_fig, label_name_dict, label_color_list, label_list, max_iter, training_data_index, testing_data_index, eval_new, prediction_list, selected_points, properties,error_message, color_list, confidence_list

def highlight_epoch_projection(context, EPOCH, predicates, TaskType,indicates):
    # TODO consider active learning setting
    error_message = ""
    start = time.time()
    all_data = get_train_test_data(context, EPOCH)
    
    labels = get_train_test_label(context, EPOCH, all_data)
    if len(indicates):
        all_data = all_data[indicates]
        labels = labels[indicates]
        
    
    # print('labels',labels)
    prediction_list = []
    # print("all_data",all_data.shape)
    all_data = all_data.reshape(all_data.shape[0],all_data.shape[1])
    if (TaskType == 'Classification'):
        # check if there is stored prediction and load
        prediction_path = os.path.join(context.strategy.data_provider.checkpoint_path(EPOCH), "custom_pred.json")
        if os.path.isfile(prediction_path):
            with open(prediction_path, "r") as f:
                predictions = json.load(f)

            for prediction in predictions:
                prediction_list.append(prediction)
                
        else:
            prediction = context.strategy.data_provider.get_pred(EPOCH, all_data).argmax(1)

            for i in range(len(prediction)):
                prediction_list.append(CLASSES[prediction[i]])
    else:
        for i in range(len(all_data)):
            prediction_list.append(0)
    n = len(prediction_list)
    labels[:n] = prediction_list
    error_message = check_labels_match_alldata(labels, all_data, error_message)
    
    embedding_2d = get_embedding(context, all_data, EPOCH)
    if len(indicates):
        embedding_2d = embedding_2d[indicates]
    print('all_data',all_data.shape,'embedding_2d',embedding_2d.shape)
    error_message = check_embedding_match_alldata(embedding_2d, all_data, error_message)
    
    training_data_number = context.strategy.config["TRAINING"]["train_num"]
    testing_data_number = context.strategy.config["TRAINING"]["test_num"]
    training_data_index = list(range(training_data_number))
    testing_data_index = list(range(training_data_number, training_data_number + testing_data_number))
    error_message = check_config_match_embedding(training_data_number, testing_data_number, embedding_2d, error_message)
    end = time.time()
    print("beforeduataion", end- start)
    # return the image of background
    # read cache if exists
   
    grid, b_fig = get_grid_bfig(context, EPOCH,embedding_2d)
    # TODO fix its structure
    eval_new = get_eval_new(context, EPOCH)
    start2 = time.time()
    print("midquestion1", start2-end)
    if TaskType == "Classification":
        print('here',labels)
        color = context.strategy.vis.get_standard_classes_color() * 255
        start3 = time.time()
        print(start3-start2)
        color = color.astype(int)

        
        label_color_list = color[labels].tolist()
       
    else:
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_data)
        labels_kmeans = kmeans.labels_
        colormap = plt.cm.get_cmap('tab10', n_clusters)
    
        colors_rgb = (colormap(np.arange(n_clusters))[:, :3] * 255).astype(int)  
        label_color_list = [colors_rgb[label].tolist() for label in labels_kmeans]
    

    start1 =time.time()
    print("midquestion2",start1-start2)
    CLASSES = np.array(context.strategy.config["CLASSES"])
    label_list = CLASSES[labels].tolist()
    label_name_dict = dict(enumerate(CLASSES))
    
    EPOCH_START = context.strategy.config["EPOCH_START"]
    EPOCH_PERIOD = context.strategy.config["EPOCH_PERIOD"]
    EPOCH_END = context.strategy.config["EPOCH_END"]
    max_iter = (EPOCH_END - EPOCH_START) // EPOCH_PERIOD + 1
    
    selected_points = get_selected_points(context, predicates, EPOCH, training_data_number, testing_data_number)
    
    properties = get_properties(context, training_data_number, testing_data_number, training_data_index, EPOCH)

    end1 = time.time()
    print("midduration", start1-end)
    print("endduration", end1-start1)
    print("EMBEDDINGLEN", len(embedding_2d))
    return embedding_2d.tolist(), grid, b_fig, label_name_dict, label_color_list, label_list, max_iter, training_data_index, testing_data_index, eval_new, prediction_list, selected_points, properties,error_message

def read_tokens_by_file(dir_path: str, n: int):
    '''Read labels from text_0.txt to text_n.txt. Only the first line is recognized as a label.'''
    labels = []
    for i in range(n):
        try:
            with open(os.path.join(dir_path, "text_{}.txt".format(i)), 'r', encoding='utf-8') as f:
                labels += [next(line for line in f).strip()]
        except Exception:
            break
    return labels

def read_file_as_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

# FIXME add cache when the content_path doesn't change
def get_umap_neighborhood_epoch_projection(content_path: str, epoch: int, predicates: list[int], indicates: list[int]):
    data_folder = os.path.join(content_path, 'Model')
    epoch_folder = os.path.join(data_folder, f'Epoch_{epoch}')

    # Read number of indices of all, comment and code
    # We only read the number of comment, then the rest are code
    # FIXME this is not a good specification of how to split comment and code
    all_indices_file = os.path.join(epoch_folder, 'index.json')
    comment_indices_file = os.path.join(epoch_folder, 'comment_index.json')
    all_idx_num = len(read_file_as_json(all_indices_file))
    cmt_idx_num = len(read_file_as_json(comment_indices_file))
    code_idx_num = all_idx_num - cmt_idx_num
    
    label_text_list = ['comment', 'code']
    labels = [0] * cmt_idx_num + [1] * code_idx_num

    # Assume there are `code_labels` and `comment_labels` folder: Read code tokens and comment tokens
    code_labels_folder = os.path.join(data_folder, 'code_labels')
    comment_labels_folder = os.path.join(data_folder, 'comment_labels')

    code_tokens = read_tokens_by_file(code_labels_folder, code_idx_num)
    comment_tokens = read_tokens_by_file(comment_labels_folder, cmt_idx_num)

    assert (len(code_tokens) == code_idx_num)
    assert (len(comment_tokens) == cmt_idx_num)

    # Read and return projections
    proj_file = os.path.join(epoch_folder, 'embedding.npy')

    proj = np.load(proj_file).tolist()

    # Read and return similarities, inter-type and intra-type
    inter_sim_file = os.path.join(epoch_folder, 'inter_similarity.npy')
    intra_sim_file = os.path.join(epoch_folder, 'intra_similarity.npy')
    
    inter_sim_top_k = np.load(inter_sim_file).tolist()
    intra_sim_top_k = np.load(intra_sim_file).tolist()
    
    # Read and return attention
    # attention_folder = os.path.join(data_folder,'aa_possim') # gcb_tokens_temp/Model/aa_possim
    # code_attention_file = os.path.join(attention_folder,'train_code_attention_aa.npy')
    # nl_attention_file = os.path.join(attention_folder,'train_nl_attention_aa.npy')
    
    # code_attention = np.load(code_attention_file).tolist()
    # nl_attention = np.load(nl_attention_file).tolist()
    
    # Read the bounding box (TODO necessary?)
    bounding_file = os.path.join(epoch_folder, 'scale.npy')
    bounding_np_array = np.load(bounding_file)
    x_min, y_min, x_max, y_max = bounding_np_array.tolist()

    result = {
        'proj': proj,
        'labels': labels,
        'label_text_list': label_text_list,
        'tokens': comment_tokens + code_tokens,
        'inter_sim_top_k': inter_sim_top_k,
        'intra_sim_top_k': intra_sim_top_k,
        # 'code_attention': code_attention,
        # 'nl_attention': nl_attention,
        'bounding': {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        }
    }

    return result

# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           BertConfig, BertForMaskedLM, BertTokenizer,
#                           GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
#                           OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                           RobertaConfig, RobertaModel, RobertaTokenizer,
#                           DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

# MODEL_CLASSES = {
#     'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
#     'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
#     'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
#     'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
#     'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
# }

import logging
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,
                 idx,

                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.idx = idx

def convert_examples_to_features(js, tokenizer, block_size):
    # code
    if 'code_tokens' in js:
        code = ' '.join(js['code_tokens'])
    else:
        code = ' '.join(js['function_tokens'])
    code_tokens = tokenizer.tokenize(code)[:block_size - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = block_size - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:block_size - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = block_size - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'], js['idx'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, block_size, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                # if i>200:
                #     break
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, block_size))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:1]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))

def update_custom_epoch_projection(context, EPOCH, predicates, TaskType,indicates, CUSTOM_PATH):
    GPU_ID = 1
    DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
    print("device", DEVICE)

    # args
    model_type = "roberta" #    parser.add_argument("--model_type", default="roberta", type=str, help="The model architecture to be fine-tuned.")
    config_name = "/home/yiming/cophi/projects/codebert-base"    # parser.add_argument("--config_name", default="", type=str, help="Optional pretrained config name or path if not the same as model_name_or_path")
    model_name_or_path =  None   # parser.add_argument("--model_name_or_path", default=None, type=str, help="The model checkpoint for weights initialization.")
    cache_dir =  "/home/yiming/cophi/projects/codebert-base"   # parser.add_argument("--cache_dir", default="/home/yiming/cophi/projects/codebert-base", type=str, help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    tokenizer_name = "/home/yiming/cophi/projects/codebert-base"    # parser.add_argument("--tokenizer_name", default="/home/yiming/cophi/projects/codebert-base", type=str, help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    do_lower_case = True   # parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    block_size = 256
    train_batch_size = 32 
    output_dir = "/home/yiming/cophi/projects/mtpnet/Text-code/NL-code-search-Adv/python/model"    
    train_batch_size = 32
    local_rank = -1
    
    
    config = config_class.from_pretrained(config_name if config_name else model_name_or_path,
                                            cache_dir=cache_dir if cache_dir else None)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir if cache_dir else None)
    if block_size <= 0:
        block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    block_size = min(block_size, tokenizer.max_len_single_sentence)
    if model_name_or_path:
        model = model_class.from_pretrained(model_name_or_path,
                                            config=config,
                                            cache_dir=cache_dir if cache_dir else None)
    else:
        model = model_class(config)

    # from Model.Model import Model
    model = Model(model, config, tokenizer)

    train_dataset = TextDataset(tokenizer, block_size, CUSTOM_PATH)
    """ Train the model """
    per_gpu_train_batch_size = train_batch_size
    train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                            output_device=local_rank,
                                                            find_unused_parameters=True)

    output_dir = os.path.join(output_dir, 'Epoch_{}'.format(EPOCH))
    model_to_save = model.module if hasattr(model, 'module') else model
    ckpt_output_path = os.path.join(output_dir, 'subject_model.pth')
    # model.load_state_dict(torch.load(ckpt_output_path, map_location=torch.device('cpu')),strict=False) 
    model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
    model.to(DEVICE)
    logger.info("Saving training feature")
    train_dataloader_bs1 = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=4,pin_memory=True)
    code_feature, nl_feature = [], []
    for batch in tqdm(train_dataloader_bs1):
        code_inputs = batch[0].to(DEVICE)
        nl_inputs = batch[1].to(DEVICE)
        model.eval()
        with torch.no_grad():
            lm_loss, code_vec, nl_vec = model(code_inputs, nl_inputs)
            # cf, nf = model.feature(code_inputs=code_inputs, nl_inputs=nl_inputs)
            code_feature.append(code_vec.cpu().detach().numpy())
            nl_feature.append(nl_vec.cpu().detach().numpy())
    code_feature = np.concatenate(code_feature, 0)
    nl_feature = np.concatenate(nl_feature, 0)
    print(code_feature.shape, nl_feature.shape)

    # TODO consider active learning setting
    error_message = ""
    start = time.time()
    all_data = code_feature
    labels = []
    prediction_list = []
    for i in range(len(all_data)):
        labels.append(0)
        prediction_list.append(0)
    if len(indicates):
        all_data = all_data[indicates]
        labels = labels[indicates]
        
    
    # print('labels',labels)
    error_message = check_labels_match_alldata(labels, all_data, error_message)
    
    embedding_2d = get_custom_embedding(context, all_data, EPOCH)
    if len(indicates):
        embedding_2d = embedding_2d[indicates]
    print('all_data',all_data.shape,'embedding_2d',embedding_2d.shape)
    error_message = check_embedding_match_alldata(embedding_2d, all_data, error_message)
    
    training_data_number = context.strategy.config["TRAINING"]["train_num"]
    testing_data_number = context.strategy.config["TRAINING"]["test_num"]
    training_data_index = list(range(training_data_number))
    testing_data_index = list(range(training_data_number, training_data_number + testing_data_number))

    end = time.time()
    print("beforeduataion", end- start)
   
    grid, b_fig = get_grid_bfig(context, EPOCH,embedding_2d)
    # TODO fix its structure
    eval_new = get_eval_new(context, EPOCH)
    start2 = time.time()
    print("midquestion1", start2-end)
    if TaskType == "Classification":
        print('here',labels)
        color = context.strategy.vis.get_standard_classes_color() * 255
        start3 = time.time()
        print(start3-start2)
        color = color.astype(int)

        
        label_color_list = color[labels].tolist()
       
    else:
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_data)
        labels_kmeans = kmeans.labels_
        colormap = plt.cm.get_cmap('tab10', n_clusters)
    
        colors_rgb = (colormap(np.arange(n_clusters))[:, :3] * 255).astype(int)  
        label_color_list = [colors_rgb[label].tolist() for label in labels_kmeans]
    

    start1 =time.time()
    print("midquestion2",start1-start2)
    CLASSES = np.array(context.strategy.config["CLASSES"])
    label_list = CLASSES[labels].tolist()
    label_name_dict = dict(enumerate(CLASSES))

    # prediction_list = []
    # # print("all_data",all_data.shape)
    # all_data = all_data.reshape(all_data.shape[0],all_data.shape[1])
    # if (TaskType == 'Classification'):
    #     # check if there is stored prediction and load
    #     prediction_path = os.path.join(context.strategy.data_provider.checkpoint_path(EPOCH), "modified_ranks.json")
    #     if os.path.isfile(prediction_path):
    #         with open(prediction_path, "r") as f:
    #             predictions = json.load(f)

    #         for prediction in predictions:
    #             prediction_list.append(prediction)
    #     else:
    #         prediction = context.strategy.data_provider.get_pred(EPOCH, all_data).argmax(1)

    #         for i in range(len(prediction)):
    #             prediction_list.append(CLASSES[prediction[i]])
    
    EPOCH_START = context.strategy.config["EPOCH_START"]
    EPOCH_PERIOD = context.strategy.config["EPOCH_PERIOD"]
    EPOCH_END = context.strategy.config["EPOCH_END"]
    max_iter = (EPOCH_END - EPOCH_START) // EPOCH_PERIOD + 1

    selected_points = get_selected_points(context, predicates, EPOCH, training_data_number, testing_data_number)
    
    properties = get_properties(context, training_data_number, testing_data_number, training_data_index, EPOCH)


    end1 = time.time()
    print("midduration", start1-end)
    print("endduration", end1-start1)
    print("EMBEDDINGLEN", len(embedding_2d))
    return embedding_2d.tolist(), grid, b_fig, label_name_dict, label_color_list, label_list, max_iter, training_data_index, testing_data_index, eval_new, prediction_list, selected_points, properties,error_message


def getVisError(context, EPOCH, TaskType):
    highlightedPointIndices = []
    all_data = get_train_test_data(context, EPOCH)

    train_data = context.train_representation_data(EPOCH)
    embedding_2d = get_embedding(context, all_data, EPOCH)
    if (TaskType == 'Classification'):
        high_pred = context.strategy.data_provider.get_pred(EPOCH, all_data).argmax(1)
        # project_embedding = context.strategy.projector.batch_project(EPOCH, train_data)
        # project_embedding1 = project_embedding
        # embed_difference = np.where(project_embedding != embedding_2d)[0]

        inv_high_dim_data = context.strategy.projector.batch_inverse(EPOCH, embedding_2d)
        inv_high_pred = context.strategy.data_provider.get_pred(EPOCH, inv_high_dim_data).argmax(1)
        # highlightedPointIndices = np.where(high_pred != inv_high_pred)[0]
        for index, (item1, item2) in enumerate(zip(inv_high_pred, high_pred)):
            if item1 != item2:
                highlightedPointIndices.append(index)
        # print(len(inv_high_dim_data))
        # print("invhighshape", inv_high_dim_data.shape)
        # print("embeddiffer", embed_difference)
        # print("embed2dlen", len(embedding_2d))
        # print("embed2dprojectlen",len(project_embedding))
        # print("invhighlen",len(inv_high_pred))

        print(high_pred)
        print(inv_high_pred)
        print(np.where(high_pred != inv_high_pred))
        print(highlightedPointIndices)
    elif (TaskType == 'Non-Classification'):
        inv_high_dim_data = context.strategy.projector.batch_inverse(EPOCH, embedding_2d)
        # todo, change train data to all data
        squared_distances = np.sum((all_data - inv_high_dim_data) ** 2, axis=1)
        squared_threshold = 1 ** 2
        highlightedPointIndices = np.where(squared_distances > squared_threshold)[0]
        highlightedPointIndices = highlightedPointIndices.tolist()
    else:
        return

    return highlightedPointIndices

	
def getContraVisChangeIndices(context_left,context_right, iterationLeft, iterationRight, method):
   
    predChangeIndices = []
    
    train_data = context_left.train_representation_data(iterationLeft)
    test_data = context_left.test_representation_data(iterationLeft)
    all_data = np.concatenate((train_data, test_data), axis=0)
    embedding_path = os.path.join(context_left.strategy.data_provider.checkpoint_path(iterationLeft), "embedding.npy")
    if os.path.exists(embedding_path):
        embedding_2d = np.load(embedding_path)
    else:
        embedding_2d = context_left.strategy.projector.batch_project(iterationLeft, all_data)
        np.save(embedding_path, embedding_2d)
    last_train_data = context_right.train_representation_data(iterationRight)
    last_test_data = context_right.test_representation_data(iterationRight)
    last_all_data = np.concatenate((last_train_data, last_test_data), axis=0)
    last_embedding_path = os.path.join(context_right.strategy.data_provider.checkpoint_path(iterationRight), "embedding.npy")
    if os.path.exists(last_embedding_path):
        last_embedding_2d = np.load(last_embedding_path)
    else:
        last_embedding_2d = context_right.strategy.projector.batch_project(iterationRight, last_all_data)
        np.save(last_embedding_path, last_embedding_2d)
    if (method == "align"):
        predChangeIndices = evaluate_isAlign(embedding_2d, last_embedding_2d)
    elif (method == "nearest neighbour"):
        predChangeIndices = evaluate_isNearestNeighbour(embedding_2d, last_embedding_2d)
    elif (method == "both"):
        predChangeIndices_align = evaluate_isAlign(embedding_2d, last_embedding_2d)
        predChangeIndices_nearest = evaluate_isNearestNeighbour(embedding_2d, last_embedding_2d)
  
        intersection = set(predChangeIndices_align).intersection(predChangeIndices_nearest)
    
        predChangeIndices = list(intersection)
    else:
        print("wrong method")
    return predChangeIndices
def getContraVisChangeIndicesSingle(context_left,context_right, iterationLeft, iterationRight, method, left_selected, right_selected):
    
    train_data = context_left.train_representation_data(iterationLeft)
    test_data = context_left.test_representation_data(iterationLeft)
    all_data = np.concatenate((train_data, test_data), axis=0)
    embedding_path = os.path.join(context_left.strategy.data_provider.checkpoint_path(iterationLeft), "embedding.npy")
    if os.path.exists(embedding_path):
        embedding_2d = np.load(embedding_path)
    else:
        embedding_2d = context_left.strategy.projector.batch_project(iterationLeft, all_data)
        np.save(embedding_path, embedding_2d)
    last_train_data = context_right.train_representation_data(iterationRight)
    last_test_data = context_right.test_representation_data(iterationRight)
    last_all_data = np.concatenate((last_train_data, last_test_data), axis=0)
    last_embedding_path = os.path.join(context_right.strategy.data_provider.checkpoint_path(iterationRight), "embedding.npy")
    if os.path.exists(last_embedding_path):
        last_embedding_2d = np.load(last_embedding_path)
    else:
        last_embedding_2d = context_right.strategy.projector.batch_project(iterationRight, last_all_data)
        np.save(last_embedding_path, last_embedding_2d)
    predChangeIndicesLeft = []
    predChangeIndicesRight = []
    predChangeIndicesLeft_Left = []
    predChangeIndicesLeft_Right = []
    predChangeIndicesRight_Left = []
    predChangeIndicesRight_Right = []
    if (method == "align"):
        predChangeIndicesLeft, predChangeIndicesRight = evaluate_isAlign_single(embedding_2d, last_embedding_2d, left_selected, right_selected)
    elif (method == "nearest neighbour"):
        predChangeIndicesLeft_Left, predChangeIndicesLeft_Right,predChangeIndicesRight_Left, predChangeIndicesRight_Right= evaluate_isNearestNeighbour_single(embedding_2d, last_embedding_2d, left_selected, right_selected)
    return predChangeIndicesLeft, predChangeIndicesRight, predChangeIndicesLeft_Left, predChangeIndicesLeft_Right, predChangeIndicesRight_Left, predChangeIndicesRight_Right

def getCriticalChangeIndices(context, curr_iteration, next_iteration):
    predChangeIndices = []
    
    all_data = get_train_test_data(context, curr_iteration)
    all_data_next = get_train_test_data(context, next_iteration)
  
    high_pred = context.strategy.data_provider.get_pred(curr_iteration, all_data).argmax(1)
    next_high_pred = context.strategy.data_provider.get_pred(next_iteration, all_data_next).argmax(1)
    predChangeIndices = np.where(high_pred != next_high_pred)[0]
    return predChangeIndices

def getConfChangeIndices(context, curr_iteration, last_iteration, confChangeInput):
    
    train_data = context.train_representation_data(curr_iteration)
    test_data = context.test_representation_data(curr_iteration)
    all_data = np.concatenate((train_data, test_data), axis=0)
    embedding_path = os.path.join(context.strategy.data_provider.checkpoint_path(curr_iteration), "embedding.npy")
    if os.path.exists(embedding_path):
        embedding_2d = np.load(embedding_path)
    else:
        embedding_2d = context.strategy.projector.batch_project(curr_iteration, all_data)
        np.save(embedding_path, embedding_2d)
    last_train_data = context.train_representation_data(last_iteration)
    last_test_data = context.test_representation_data(last_iteration)
    last_all_data = np.concatenate((last_train_data, last_test_data), axis=0)
    last_embedding_path = os.path.join(context.strategy.data_provider.checkpoint_path(last_iteration), "embedding.npy")
    if os.path.exists(last_embedding_path):
        last_embedding_2d = np.load(last_embedding_path)
    else:
        last_embedding_2d = context.strategy.projector.batch_project(last_iteration, last_all_data)
        np.save(last_embedding_path, last_embedding_2d)
    high_pred = context.strategy.data_provider.get_pred(curr_iteration, all_data)
    last_high_pred = context.strategy.data_provider.get_pred(last_iteration, last_all_data)
    high_conf = softmax(high_pred, axis=1)
    last_high_conf = softmax(last_high_pred, axis=1)
    # get class type with highest prob
    high_pred_class = high_conf.argmax(axis=1)
    last_high_pred_class = last_high_conf.argmax(axis=1)
    same_pred_indices = np.where(high_pred_class == last_high_pred_class)[0]
    print("same")
    print(same_pred_indices)
    # get
    conf_diff = np.abs(high_conf[np.arange(len(high_conf)), high_pred_class] - last_high_conf[np.arange(len(last_high_conf)), last_high_pred_class])
    print("conf")
    print(conf_diff)
    significant_conf_change_indices = same_pred_indices[conf_diff[same_pred_indices] > confChangeInput]
    print("siginificant")
    print(significant_conf_change_indices)
    return significant_conf_change_indices

def add_line(path, data_row):
    """
    data_row: list, [API_name, username, time]
    """
    now_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    data_row.append(now_time)
    with open(path, "a+") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)