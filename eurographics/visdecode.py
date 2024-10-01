from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from huggingface_hub import snapshot_download
import os
from tqdm import tqdm
import numpy as np
import Levenshtein
from colors import *

MAX_LENGTH = 600

def load_model(owner, model_name, device):
    
    path = os.getcwd() + "/models/"

    processor = None
    model = None

    try:
        model = Pix2StructForConditionalGeneration.from_pretrained(path + model_name).to(device)
    except:
        snapshot_download(repo_id = owner + "/" + model_name, local_dir = path + model_name)
        model = Pix2StructForConditionalGeneration.from_pretrained(path + model_name)

    try:
        processor = AutoProcessor.from_pretrained(path + "matcha-base")
    except:
        snapshot_download(repo_id = "google/matcha-base", local_dir = path + "matcha-base")
        processor = AutoProcessor.from_pretrained(path + "matcha-base")

    processor.image_processor.is_vqa = False

    return processor, model

def generate(processor, model, images, device):

    outputs = []

    for image in tqdm(images):

        model.eval()
        inputs = processor(images = image, return_tensors = "pt", max_patches = 1024).to(device)

        tokens = model.generate(flattened_patches = inputs.flattened_patches, attention_mask = inputs.attention_mask, max_length = MAX_LENGTH)
        output = processor.batch_decode(tokens, skip_special_tokens = True)[0]

        outputs.append(output)

    return outputs

def multiclass_confusion_matrix(samples, classes):
    
    n = len(classes)
    confusion_mat = np.zeros((n,n))

    for pred_class, gt_class in samples:

        i = classes.index(gt_class)
        j = classes.index(pred_class)

        confusion_mat[i,j] += 1

    return confusion_mat

def f1_score(multiclass_confusion_mat, classes):
    
    n = multiclass_confusion_mat.shape[0]
    scores = {}

    for i in range(len(classes)):
        
        TP, FN, FP, = 0.0, 0.0, 0.0

        for col in range(n):

            if i == col: 
                TP += multiclass_confusion_mat[i, col]
            else:
                FN += multiclass_confusion_mat[i, col]
                    
        for row in range(n):
            if i != row: FP += multiclass_confusion_mat[row, i]

        precision   = np.round(TP / (TP + FP), 2) if TP + FP > 0 else 0 
        recall      = np.round(TP / (TP + FN), 2) if TP + FN > 0 else 0

        f1 = np.round(2 * (precision * recall) / (precision + recall), 2) if precision + recall > 0 else 0

        scores[classes[i]] = f1

    return scores

def re_score(samples):
    
    scores = [Levenshtein.ratio(sample[0], sample[1]) for sample in samples]
    mean_score = np.round(np.mean(scores), 2)

    return mean_score

def rms_score(samples):
    
    scores = []

    for sample in samples:

        0

    return np.round(np.mean(scores), 2)

def dict_mean(dict):

    dict_values = [v for v in dict.values() if v is not None]
    return (sum(dict_values) / len(dict_values)) if len(dict_values) > 0 else None

def extract_from_vegas(vegas, gt_vegas):

    output = {"marks":[], 
                    "x": {"types":[], "names":[]}, 
                    "y": {"types":[], "names":[]},
                    "data":[]}
    
    for vega, gt_vega in zip(vegas, gt_vegas):

        try:
            # fill vega_outputs dict with pairs (out, gt) for mark, types, names...
            
            mark_type = (vega["mark"], gt_vega["mark"])

            x_type = (vega["encoding"]["x"]["type"], gt_vega["encoding"]["x"]["type"])
            y_type = (vega["encoding"]["y"]["type"], gt_vega["encoding"]["y"]["type"])

            x_name = (vega["encoding"]["x"]["field"], gt_vega["encoding"]["x"]["field"])
            y_name = (vega["encoding"]["y"]["field"], gt_vega["encoding"]["y"]["field"])

            data = (vega["data"]["values"], gt_vega["data"]["values"])

            # ----------------------------------

            output["marks"].append(mark_type)

            output["x"]["types"].append(x_type)
            output["y"]["types"].append(y_type)

            output["x"]["names"].append(x_name)
            output["y"]["names"].append(y_name)

            output["data"].append(data)

        except:
            print("error")

    return output

def compute_metrics(vegas, gt_vegas, average = False):
    
    input = extract_from_vegas(vegas, gt_vegas)

    # mark-type score

    mark_classes = ["bar","line","circle"]
    marks = input["marks"]

    marks_confusion_mat = multiclass_confusion_matrix(marks, mark_classes)
    mark_score = f1_score(marks_confusion_mat, mark_classes)

    # var types score

    var_types_classes = ["quantitative", "temporal", "nominal", "ordinal"]

    x_types = input["x"]["types"]
    y_types = input["y"]["types"]

    x_types_confusion_mat = multiclass_confusion_matrix(x_types, var_types_classes)
    y_types_confusion_mat = multiclass_confusion_matrix(y_types, var_types_classes)

    x_type_score = f1_score(x_types_confusion_mat, var_types_classes)
    y_type_score = f1_score(y_types_confusion_mat, ["quantitative"])

    # var-names score

    x_names = input["x"]["names"]
    y_names = input["y"]["names"]

    x_name_score = re_score(x_names)
    y_name_score = re_score(y_names)

    # data score

    data = input["data"]
    data_score = rms_score(data)

    if average:

        mark_avg_score = np.round(dict_mean(mark_score), 2)

        var_type_avg_score = np.round( dict_mean({"x_type": dict_mean(x_type_score), "y_type": dict_mean(y_type_score)}), 2)
        var_name_avg_score = np.round( np.mean([x_name_score, y_name_score]), 2)

        return {"mark_type": mark_avg_score, "var_type": var_type_avg_score, "var_name": var_name_avg_score}

    return {"mark_type": mark_score, "x_type": x_type_score, "y_type": y_type_score, "x_name": x_name_score, "y_name": y_name_score}

def text_to_vega(texts, ret_status = False):

    vegas = []
    status = []

    for text in texts:

        try: 
            vegas.append(json.loads(text.replace("'",'"')))
            status.append(True)
        except: 
            status.append(False)

    if ret_status: 
        
        error_rate = 100 - np.round(np.mean(status) * 100, 2)
        print(magenta("|"), "JSON to Vega conversion error rate:", red(str(error_rate)), red("%"), magenta("|"))

        return vegas, status
    
    return vegas

def eval(texts, gt_texts):

    vegas, status = text_to_vega(texts, ret_status = True)

    gt_vegas = text_to_vega(gt_texts)
    gt_vegas = [item for item, cond in zip(gt_vegas, status) if cond]

    metrics = compute_metrics(vegas, gt_vegas, average = False)

    # compute average metrics 

    mark_type_score = np.round(dict_mean(metrics["mark_type"]), 2)

    x_type_score = np.round(dict_mean(metrics["x_type"]), 2)
    y_type_score = np.round(dict_mean(metrics["y_type"]), 2)

    x_name_score = metrics["x_name"]
    y_name_score = metrics["y_name"]

    print(bold(magenta("----------------------------------------------------- EVALUATION -------------------------------------------------------")))
    print(magenta("|"), bold(cyan("MARK-TYPE")), ":", mark_type_score, magenta("|"), bold(cyan("X-TYPE")), ":", x_type_score, magenta("|"), bold(cyan("Y-TYPE")), ":", y_type_score, magenta("|"), bold(cyan("X-NAME")), ":", x_name_score, magenta("|"), bold(cyan("Y-NAME")), ":", y_name_score, magenta("|"))
    print(bold(magenta("------------------------------------------------------------------------------------------------------------------------\n")))

    for i, text in enumerate(texts):

        print(bold(green(gt_texts[i])))
        print(text if status[i] else red(text), "\n")

    print(bold(magenta("------------------------------------------------------------------------------------------------------------------------")))
    print(magenta("|"), bold(cyan("MARK-TYPE")), ":", mark_type_score, magenta("|"), bold(cyan("X-TYPE")), ":", x_type_score, magenta("|"), bold(cyan("Y-TYPE")), ":", y_type_score, magenta("|"), bold(cyan("X-NAME")), ":", x_name_score, magenta("|"), bold(cyan("Y-NAME")), ":", y_name_score, magenta("|"))
    print(bold(magenta("------------------------------------------------------------------------------------------------------------------------\n")))

    return metrics