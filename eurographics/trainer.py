import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import login
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from Levenshtein import distance as levenshtein_distance
import wandb
import json

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

BOLD = "\033[1m"

EPOCHS = 50
EVAL_STEP = 5
MODEL = "visdecode_D"
TRAIN_MODEL = "google/matcha-base"
MAX_PATCHES = 1024
UPLOAD_METRICS = True
MAX_ACCURACY_TEST = 0.3
MAX_LENGTH = 600

login(token = "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")

seed = 14895215085708117999
torch.manual_seed(seed)

visdecode_dataset_train = load_dataset("martinsinnona/visdecode", split = "train")
visdecode_dataset_test = load_dataset("martinsinnona/plotqa", split = "validation")

visdecode_dataset_test2 = load_dataset("martinsinnona/visdecode_web", split = "test")
visdecode_dataset_test3 = load_dataset("martinsinnona/plotqa", split = "test")

print("\nTRAIN:\n", visdecode_dataset_train, "\nTEST:\n", visdecode_dataset_test, "\nWEB:\n", visdecode_dataset_test2,"\n")

processor = AutoProcessor.from_pretrained("google/matcha-base")
processor.image_processor.is_vqa = False

model_name = ("martinsinnona/" + MODEL) if EPOCHS == -1 else TRAIN_MODEL
model = Pix2StructForConditionalGeneration.from_pretrained(model_name)

print("\n --------> using model base:", model_name, "<--------\n")

if UPLOAD_METRICS:

    wandb.login(key = "451637d95c22df4568c6f5a268e37071bc14547b")
    wandb.init(
        project="visdecode", 
        entity="martinsinnona", 
        config = {}
    )


def compare_strings(str1, str2):

    dmax = max(len(str1), len(str2), 1)
    d = (levenshtein_distance(str1,str2))

    return 1 - d / dmax

def get_mark_type(str):
    
    start = str.find("<mark>")
    end = str.find("</mark>")
    
    if start != -1 and end != -1: return str[start+6:end]
    return ""

def get_var_types(str):
    
    start1 = str.find("<type>")
    end1 = str.find("</type>")
    
    if start1 != -1 and end1 != -1: 
        
        start2 = str.find("<type>", end1+1)
        end2 = str.find("</type>", end1+1)
        
        return str[start1+6:end1], str[start2+6:end2]
    
    return "",""

def get_var_names(str):
    
    start1 = str.find("<field>")
    end1 = str.find("</field>")
    
    if start1 != -1 and end1 != -1: 
        
        start2 = str.find("<field>", end1+1)
        end2 = str.find("</field>", end1+1)
        
        return str[start1+7:end1], str[start2+7:end2]
    
    return "",""

class ImageCaptioningDataset(Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text = "", return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)

        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item["text"]

        return encoding

def collator(batch):

    new_batch = {"flattened_patches":[], "attention_mask":[]}
    texts = [item["text"] for item in batch]

    text_inputs = processor(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=MAX_LENGTH)

    new_batch["labels"] = text_inputs.input_ids

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch

train_dataset = ImageCaptioningDataset(visdecode_dataset_train, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)

def compute_accuracies(results, output_raw = False):
    
    accuracies_mark_type = []
    accuracies_var_types = []

    for res in results:

        accuracy_mark_type = get_mark_type(res[0]) == get_mark_type(res[1])
        accuracy_var_types = get_var_types(res[0]) == get_var_types(res[1])

        accuracies_mark_type.append(accuracy_mark_type)
        accuracies_var_types.append(accuracy_var_types)

    accuracy_mark_type = np.round(np.mean(accuracies_mark_type), 2)
    accuracy_var_types = np.round(np.mean(accuracies_var_types), 2)
    
    if output_raw: return accuracies_mark_type, accuracies_var_types
    return accuracy_mark_type, accuracy_var_types

def compute_accuracies2(results, output_raw = False, print_output = False):
    
    accuracies_mark_type = []

    accuracies_var_types_x = []
    accuracies_var_types_y = []

    accuracies_var_names_x = []
    accuracies_var_names_y = []

    accuracies_structure = []

    for res in results:

        mark_type_acc, var_type_x_acc, var_type_y_acc, var_name_x_acc, var_name_y_acc = 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            
            input_vega = json.loads(res[0].replace("'",'"'))
            gt_vega = json.loads(res[1].replace("'",'"'))

            mark_type_acc = (input_vega["mark"] == gt_vega["mark"]) * 1.00

            var_type_x_acc = (input_vega["encoding"]["x"]["type"] == gt_vega["encoding"]["x"]["type"]) * 1.00
            var_type_y_acc = (input_vega["encoding"]["y"]["type"] == gt_vega["encoding"]["y"]["type"]) * 1.00

            var_name_x_acc = compare_strings(input_vega["encoding"]["x"]["field"], gt_vega["encoding"]["x"]["field"])
            var_name_y_acc = compare_strings(input_vega["encoding"]["y"]["field"], gt_vega["encoding"]["y"]["field"])

            accuracies_mark_type.append(mark_type_acc)

            accuracies_var_types_x.append(var_type_x_acc)
            accuracies_var_types_y.append(var_type_y_acc)

            accuracies_var_names_x.append(var_name_x_acc)
            accuracies_var_names_y.append(var_name_y_acc)

            accuracies_structure.append(1)

            # -----------------------------------------------

            mark_type_acc = np.round(mark_type_acc, 2)

            var_type_x_acc = np.round(var_type_x_acc, 2)
            var_type_y_acc = np.round(var_type_y_acc, 2)

            var_name_x_acc = np.round(var_name_x_acc, 2)
            var_name_y_acc = np.round(var_name_y_acc, 2)

        except:
            accuracies_structure.append(0)

        if print_output:
            
            color1 = RESET if accuracies_structure[-1] == 1 else RED

            print(RESET + "---------------------------------------------------------------------------------------------------------")
            print(RESET, BOLD + "(" + str(res[2]) + ")   " + RESET + "MARK:", CYAN, mark_type_acc, RESET, "TYPE-X:", CYAN, var_type_x_acc, RESET, "TYPE-Y:", CYAN, var_type_y_acc, RESET, "NAME-X:", CYAN, var_name_x_acc, RESET, "NAME-Y:", CYAN, var_name_y_acc, "\n")

            print(GREEN, res[0])
            print(color1, res[1], "\n")

    if len(accuracies_mark_type) == 0: accuracies_mark_type.append(0)
    
    if len(accuracies_var_types_x) == 0: accuracies_var_types_x.append(0)
    if len(accuracies_var_types_y) == 0: accuracies_var_types_y.append(0)
        
    if len(accuracies_var_names_x) == 0: accuracies_var_names_x.append(0)
    if len(accuracies_var_names_y) == 0: accuracies_var_names_y.append(0)
            
    accuracy_mark_type = np.round(np.mean(accuracies_mark_type),2)

    accuracy_var_types_x = np.round(np.mean(accuracies_var_types_x),2)
    accuracy_var_types_y = np.round(np.mean(accuracies_var_types_y),2)
    
    accuracy_var_names_x = np.round(np.mean(np.array(accuracies_var_names_x)),2)
    accuracy_var_names_y = np.round(np.mean(np.array(accuracies_var_names_y)),2)

    accuracy_structure = np.round(np.mean(accuracies_structure), 2)

    if output_raw: return accuracies_mark_type, accuracies_var_types_x, accuracies_var_types_y, accuracies_var_names_x, accuracies_var_names_y, accuracies_structure
    return accuracy_mark_type, accuracy_var_types_x, accuracy_var_types_y, accuracy_var_names_x, accuracy_var_names_y, accuracy_structure
    
def eval_model(dataset, print_output = False, raw_output = False):
    
    results = []
    
    for i,data in enumerate(dataset):
        
        image = data["image"]

        model.eval()
        inputs = processor(images=image, return_tensors="pt", max_patches=MAX_PATCHES).to(device)

        flattened_patches = inputs.flattened_patches
        attention_mask = inputs.attention_mask

        generated_ids = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=MAX_LENGTH)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        result = (data["text"], generated_caption, i)
        results.append(result)

        _ = compute_accuracies2([result], raw_output, print_output)
            
    acc_mark_type, acc_var_types_x, acc_var_types_y, acc_var_names_x, acc_var_names_y, acc_struct = compute_accuracies2(results, raw_output, False)
    
    print(RESET, "\n---------------- RESULTS --------------------\n")
    
    print("accuracy mark type :", acc_mark_type,"\n")

    print("accuracy var types (x) :", acc_var_types_x)
    print("accuracy var types (y) :", acc_var_types_y,"\n")

    print("accuracy var names (x) :", acc_var_names_x)
    print("accuracy var names (y) :", acc_var_names_y,"\n")

    print("accuracy structure :", acc_struct,"\n")

    print("\n---------------------------------------------\n")
    
    #return acc_mark_type, acc_var_types
    return acc_mark_type, acc_var_types_x, acc_var_types_y, acc_var_names_x, acc_var_names_y, acc_struct

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5) 

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

losses = []
accuracies_train = []
accuracies_test = []

# ---------------------------------------- TRAINING ----------------------------------------------

for epoch in range(EPOCHS + 1):

    if epoch == 0: start_time = time.time()
    print("Epoch: ", epoch)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)
    processor.image_processor.is_vqa = False

    for idx, batch in enumerate(train_dataloader):

        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model(flattened_patches = flattened_patches,
                    attention_mask = attention_mask,
                    labels = labels)

        # ------------

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim = -1)
        
        token_ids = torch.argmax(probs, dim = -1)
        tokens = processor.batch_decode(token_ids, skip_special_tokens=True)[0]

        #print(tokens)

        # ------------

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
    if epoch == 0:
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        predicted_time = (elapsed_time * (EPOCHS+1)) / 3600

        print("\nAproximadamente quedan: " + str(round(predicted_time,2)) + " horas.\n")
    
    if epoch % EVAL_STEP == 0:
        
        #acc_mark_type, acc_var_types  = eval_model(visdecode_dataset_test, print_output = True)
        #acc_mark_type_plotqa, acc_var_types_plotqa  = eval_model(visdecode_dataset_test3)

        acc_mark_type, acc_var_types_x, acc_var_types_y, acc_var_names_x, acc_var_names_y, acc_structure = eval_model(visdecode_dataset_test, print_output = True)

        accuracy_test = min(acc_mark_type, acc_var_types_x, acc_var_types_y, acc_var_names_x, acc_var_names_y)   # CRITERIO PARA PUSHEAR A HUGGING FACE <----------------------------------------------------------
            
        if accuracy_test >= MAX_ACCURACY_TEST:
            
            model.push_to_hub(MODEL)
            MAX_ACCURACY_TEST = accuracy_test

        accuracies_test.append(accuracy_test)
        print("\naccuracies test: ", accuracies_test,"\n")

        if UPLOAD_METRICS: 
            
            wandb.log({"mark_type": acc_mark_type, 
                       
                       "var_types_x": acc_var_types_x, 
                       "var_types_y": acc_var_types_y, 
                       
                       "var_names_x": acc_var_names_x, 
                       "var_names_y": acc_var_names_y})

    losses.append(loss.cpu().detach().numpy().item())
    if UPLOAD_METRICS: wandb.log({"loss": loss.cpu().detach().numpy().item()})

eval_model(visdecode_dataset_test, print_output = True)
eval_model(visdecode_dataset_test2, print_output = True)
eval_model(visdecode_dataset_test3, print_output = True)

if UPLOAD_METRICS: wandb.finish()

