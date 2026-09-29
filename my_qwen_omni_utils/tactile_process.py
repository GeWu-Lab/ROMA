import numpy as np
from PIL import Image
from torchvision import transforms
import torch

tactile_transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize(size=(224, 224), antialias=True),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
])

to_sensor_transform = transforms.ToTensor()

tactile_offset = 130.0 / 255.0

def process_tactile_info(conversations):
    """
    Read and process tactile info

    Support dict keys:

    type = tactile
    """
    tactile_info = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation: 
            #print(message)
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "tactile":
                    tactile_img = []
                    tactile_bg = None
                    if "tactile" in ele:
                        assert isinstance(ele["tactile"], list), "tactile must be a list"
                        assert len(ele["tactile"]) == 2, "tactile must be a list of length 2"
                        for tactile_path in ele["tactile"]:
                            tactile_img.append(to_sensor_transform(Image.open(tactile_path).convert('RGB')))
                    if "tactile_bg" in ele:
                        tactile_bg = to_sensor_transform(Image.open(ele["tactile_bg"]).convert('RGB'))
                    
                    tactile_img = torch.stack(tactile_img, dim=0)
                    
                    if tactile_bg is not None:
                        tactile_img = tactile_img - tactile_bg
                        tactile_img = tactile_img + tactile_offset
                        tactile_img = torch.clamp(tactile_img, 0, 1)
                        tactile_img = tactile_transform(tactile_img)
                    else:
                        tactile_img = tactile_transform(tactile_img)
                    
                    tactile_info.append(tactile_img)

    if len(tactile_info) == 0:
        tactile_info = None
    return tactile_info

                
