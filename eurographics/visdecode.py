from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from huggingface_hub import snapshot_download

def load_model(owner, model_name, device):
    
    path = "/mnt/disk2/msinnona/models/" + model_name

    processor = None
    model = None

    try:
        processor = AutoProcessor.from_pretrained("google/matcha-base")
        model = Pix2StructForConditionalGeneration.from_pretrained(path).to(device)
    except:

        snapshot_download(repo_id = owner + "/" + model_name, local_dir=path)

        processor = AutoProcessor.from_pretrained("google/matcha-base")
        model = Pix2StructForConditionalGeneration.from_pretrained(path)

    processor.image_processor.is_vqa = False

    return processor, model

def generate(processor, model, image, device):

    model.eval()
    inputs = processor(images = image, return_tensors = "pt", max_patches = 1024).to(device)

    tokens = model.generate(flattened_patches = inputs.flattened_patches, attention_mask = inputs.attention_mask, max_length = 600)
    output = processor.batch_decode(tokens, skip_special_tokens = True)[0]

    return output

def eval_model(processor, model, image, device):

    output = generate(processor, model, image, device)
    return output