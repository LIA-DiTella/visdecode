import torch
from colors import *
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import wandb
import visdecode
from visdecode import *

MODEL = "a"
TRAIN_MODEL = "matcha-base"

UPLOAD_METRICS = True

LR = 1e-7 * 5
EPOCHS = 100
EVAL_STEP = 5
BEST_ACCURACY = 0.3
MAX_LENGTH = 600

UPLOAD_METRICS = UPLOAD_METRICS and EPOCHS != -1

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
login(token = "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")

dataset_train = load_dataset("martinsinnona/visdecode", split = "train")

dataset_val1 = load_dataset("martinsinnona/visdecode", split = "validation")
dataset_val2 = load_dataset("martinsinnona/plotqa", split = "validation")

dataset_test1 = load_dataset("martinsinnona/plotqa", split = "test")
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

if UPLOAD_METRICS:

    wandb.login(key = "451637d95c22df4568c6f5a268e37071bc14547b")
    wandb.init(
        project="visdecode", 
        entity="martinsinnona", 
        config = {}
    )

class ImageCaptioningDataset(Dataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text = "", return_tensors="pt", add_special_tokens=True, max_patches=1024)

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

train_dataset = ImageCaptioningDataset(dataset_train, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)
optimizer = torch.optim.AdamW(model.parameters(), lr = LR) 

model.to(device)
model.train()

losses = []

for epoch in range(EPOCHS + 1):

    for idx, batch in enumerate(train_dataloader):

        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model(flattened_patches = flattened_patches, attention_mask = attention_mask, labels = labels)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim = -1)
        
        token_ids = torch.argmax(probs, dim = -1)
        tokens = processor.batch_decode(token_ids, skip_special_tokens=True)[0]

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    print(bold(cyan("Epoch :")), epoch, bold(green(" | Loss :")), loss.item())
    losses.append(loss.cpu().detach().numpy().item())

    # -------------------------------------

    metrics_val1 = eval_model(processor, model, dataset_val1, device)
    metrics_val2 = eval_model(processor, model, dataset_val2, device)

    if UPLOAD_METRICS: 
        wandb.log({"val1_y_name": metrics_val1["y_name"], "val2_y_name": metrics_val2["y_name"], "loss": loss.cpu().detach().numpy().item()})

eval_model(processor, model, dataset_test1, device)
eval_model(processor, model, dataset_test2, device)

if UPLOAD_METRICS: wandb.finish()