from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from huggingface_hub import snapshot_download
import os
from tqdm import tqdm

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