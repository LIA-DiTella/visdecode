from huggingface_hub import login
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from huggingface_hub import snapshot_download

def load(owner, model_name, device):
    
    path = "/mnt/disk2/msinnona/models/" + model_name

    processor = None
    model = None

    try:
        processor = AutoProcessor.from_pretrained(path)
        model = Pix2StructForConditionalGeneration.from_pretrained(path).to(device)
    except:

        snapshot_download(repo_id = owner + "/" + model_name, local_dir=path)

        processor = AutoProcessor.from_pretrained(path)
        model = Pix2StructForConditionalGeneration.from_pretrained(path)

    processor.image_processor.is_vqa = False

    return processor, model