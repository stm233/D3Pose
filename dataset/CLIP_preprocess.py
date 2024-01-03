import os
import numpy as np
import torch
from PIL import Image
import json
from transformers import CLIPProcessor,  CLIPVisionModel
import time

if __name__ == '__main__':

    # defining path
    root_path = '/media/hongji/4T/Downloads/H36M/output_feat'

    # GTpath = '/media/hongji/4T/Downloads/H36M/annotations_smpl'
    # json_path = '/media/hongji/4T/Downloads/H36M_original/annotations.zip/annotations/Human36M_subject1_data.json'
    # # '/media/hongji/4T/Downloads/H36M_annot_smpl/Human36M_subject1_SMPLX_NeuralAnnot.json'
    # with open(json_path, 'r') as file:
    #     data = json.load(file)

    # loading CLIP
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to('cuda')

    # Walk through each directory and subdirectory
    for root, dirs, files in os.walk(root_path):
        if 'clip' in root:
            # save_tensor = []
            imageholder = []
            for frame in files:
                if frame.lower().endswith((".jpg", ".jpeg")):  # Check if the file is a JPEG image

                    frame_path = os.path.join(root, frame)
                    frame = Image.open(frame_path)
                    # CLIP_inputs = processor(images=frame, return_tensors="pt")
                    # outputs = model(**CLIP_inputs.to('cuda'))
                    # last_hidden_state = outputs.last_hidden_state

                    imageholder.append(frame)

            CLIP_inputs = processor(images=imageholder, return_tensors="pt")
            outputs = model(**CLIP_inputs.to('cuda'))
            last_hidden_state = outputs.last_hidden_state

            save_path = os.path.join(root, "image_feat.pt")
            # output = torch.cat(save_tensor, dim=0)
            torch.save(last_hidden_state.detach(), save_path)

            print('saving pt file at', root, time.time())






