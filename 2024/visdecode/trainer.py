import torch
from colors import *
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import wandb
import visdecode
from visdecode import *
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import gc
from datetime import datetime
import matplotlib.pyplot as plt

DEVICE_ID = 0

MODEL = "plotQA_simple_10"
TRAIN_MODEL = "matcha-base"
COMMENT = "PlotQA testing aumenta de tamaño (1000). Quiero generar un modelo base para comparar"

UPLOAD_METRICS = True

LR = 1e-5
EPOCHS = 100
EVAL_STEP = 5

MAX_LENGTH = 50
visdecode.MAX_LENGTH = MAX_LENGTH
visdecode.VAR_TYPE_CLASSES = ["quantitative", "temporal", "nominal"]
BATCH_SIZE = 16

MAX_SCORE = 0.1
MAX_EPOCH = -1
UPLOAD_METRICS = UPLOAD_METRICS and EPOCHS != -1
torch.manual_seed(42)
torch.cuda.manual_seed(42)


device = "cuda:" + str(DEVICE_ID) if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(DEVICE_ID)
wandb.login(key = "451637d95c22df4568c6f5a268e37071bc14547b")
login(token = "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")

dataset_train = load_dataset("martinsinnona/plotqa_simple_2", split = "train")

dataset_val1 = load_dataset("martinsinnona/visdecode_simple", split = "validation")
dataset_val2 = load_dataset("martinsinnona/plotqa_simple_2", split = "validation")

dataset_test1 = load_dataset("martinsinnona/plotqa_simple_2", split = "test")
dataset_test2 = load_dataset("martinsinnona/visdecode_web", split = "test")

print(bold(green("\n[Train] :")), len(dataset_train))

print(bold(green("[Val] :")), len(dataset_val1))
print(bold(green("[Val #2] :")), len(dataset_val2))

print(bold(green("[Test #1] :")), len(dataset_test1))
print(bold(green("[Test #2] :")), len(dataset_test2))
model_name = MODEL if EPOCHS == -1 else TRAIN_MODEL
owner = "martinsinnona" if EPOCHS == -1 else "google"

print("> Using model: ", bold(red(model_name)))
processor, model = visdecode.load_model(owner, model_name, device)
class ImageCaptioningDataset(Dataset):

    def __init__(self, dataset, processor, transform = None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        image = item["image"].convert("RGBA")

        if self.transform: 
             
            image = self.transform(image)

            white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(white_background, image)

            image = image.convert("RGB")

        encoding = self.processor(images=image, text = "", return_tensors="pt", add_special_tokens=True, max_patches=1024)

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
transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomRotation(degrees = (0, 360), expand = True)], p=0),
])
train_dataset = ImageCaptioningDataset(dataset_train, processor, transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collator)
optimizer = torch.optim.AdamW(model.parameters(), lr = LR) 
model.to(device)

0
if UPLOAD_METRICS:

    wandb.init(
        project = "visdecode",
        name = MODEL, 
        entity = "martinsinnona", 
        config = {"learning_rate": LR,
                  "epochs": EPOCHS,
                  "max_length": MAX_LENGTH,
                  "batch_size": BATCH_SIZE,
                  "comment": COMMENT}
    )

losses = []

for epoch in range(EPOCHS + 1):

    model.train()

    for idx, batch in enumerate(train_dataloader):

        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model.forward(flattened_patches = flattened_patches, attention_mask = attention_mask, labels = labels)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim = -1)
        
        token_ids = torch.argmax(probs, dim = -1)
        tokens = processor.batch_decode(token_ids, skip_special_tokens=True)[0]

        loss = outputs.loss
        loss.backward()     

        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.empty_cache()  
        gc.collect() 

    print(bold(cyan("Epoch :")), epoch, bold(green(" | Loss :")), loss.item())

    losses.append(loss.item())

    # -------------------------------------

    if epoch % EVAL_STEP == 0:

        metrics_val1 = eval_model(processor, model, dataset_val1, device, vega_structure = False)   # visdecode dataset validation
        metrics_val2 = eval_model(processor, model, dataset_val2, device, vega_structure = False)   # plotQA dataset validation

        if dict_mean(metrics_val2) > MAX_SCORE:

            model.push_to_hub(MODEL)
            MAX_SCORE = dict_mean(metrics_val2)
            MAX_EPOCH = epoch

        if UPLOAD_METRICS: 

            wandb.log({
                
                    "mark_type_val_1":      metrics_val1["mark_type"],
                    "x_type_val_1":         metrics_val1["x_type"], 
                    "y_type_val_1":         metrics_val1["y_type"], 
                    "x_name_val_1":         metrics_val1["x_name"],  
                    "y_name_val_1":         metrics_val1["y_name"],
                    "struct_error_val_1":   metrics_val1["struct_error"], 

                    "mark_type_val_2":      metrics_val2["mark_type"],
                    "x_type_val_2":         metrics_val2["x_type"], 
                    "y_type_val_2":         metrics_val2["y_type"],
                    "x_name_val_2":         metrics_val2["x_name"],
                    "y_name_val_2":         metrics_val2["y_name"], 
                    "struct_error_val_2":   metrics_val2["struct_error"],

                    "epoch": epoch,
                    "max_epoch": MAX_EPOCH,
                    "loss": loss.item()})
            
if UPLOAD_METRICS: wandb.finish()
def parameters_change(parameters, new_parameters):
    
    means = []

    with torch.no_grad():

        for p, new_p in zip(parameters, new_parameters):
            
            diff = torch.abs(p[0] - new_p[0])
            means.append(torch.mean(diff).item())

    return means

_, model_matcha = visdecode.load_model("google", "matcha-base", device)
means = parameters_change(model_matcha.parameters(), model.parameters())

plt.figure(figsize=(20,5))
plt.bar(np.arange(len(means)), means)

plt.title("mean change of weights")

plt.xlabel("weights layer index")
plt.ylabel("mean change")

#plt.savefig("mean_weights_change.png", dpi=300, bbox_inches='tight')
eval_model(processor, model, dataset_test1, device, vega_structure = False)
#eval_model(processor, model, dataset_test2, device, vega_structure = False)