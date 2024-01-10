import os
import numpy as np
import torch
from PIL import Image
import json
from transformers import CLIPProcessor,  CLIPVisionModel
import time


def process_H36M(model, path):

    # Walk through each directory and subdirectory
    for root, dirs, files in os.walk(h36m_root_path):
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


def process_3DPW(model, path):
    # Walk through each directory and subdirectory

    for segment in os.listdir(path):
        segment_path = os.path.join(path, segment)
        for vid in os.listdir(segment_path):
            vid_path = os.path.join(segment_path, vid)
            for clip in os.listdir(vid_path):
                clip_path = os.path.join(vid_path, clip)
                imageholder = []
                for frame in os.listdir(clip_path):
                    if frame.lower().endswith((".jpg", ".jpeg")):  # Check if the file is a JPEG image

                        frame_path = os.path.join(clip_path, frame)
                        frame = Image.open(frame_path)
                        imageholder.append(frame)

                CLIP_inputs = processor(images=imageholder, return_tensors="pt")
                outputs = model(**CLIP_inputs.to('cuda'))
                last_hidden_state = outputs.last_hidden_state

                save_path = os.path.join(clip_path, "image_feat.pt")
                # output = torch.cat(save_tensor, dim=0)
                torch.save(last_hidden_state.detach(), save_path)

                print('saving pt file at', clip_path)


if __name__ == '__main__':

    # defining path
    h36m_root_path = '/media/hongji/Expansion/Human3.6M/processed_data'
    dpw_path = '/media/hongji/Expansion/3DPW/processed_data'

    # loading CLIP
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to('cuda')

    # process_H36M(model, h36m_root_path)
    process_3DPW(model, dpw_path)








