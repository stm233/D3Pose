import os

import torch
from PIL import Image
import json
import shutil
import numpy as np
from transformers import CLIPProcessor,  CLIPVisionModel
import time


def process_images_and_gt(src_img_directory, dest_directory, gt_directory, camera, model):
    for folder_name in os.listdir(src_img_directory):
        # Extract subject, action, subaction from folder name
        parts = folder_name.split('_')
        subject, action, subaction = parts[1], parts[3], parts[5]
        subject = str(int(subject))
        action = str(int(action))  # Convert action index to integer and back to string to remove leading zero
        subaction = str(int(subaction))  # Same for subaction index
        subject_list = ['s_01', 's_05', 's_06', 's_07', 's_08', 's_09', 's_11', ]

        # Check if the folder is for the first camera angle
        if camera in folder_name:
            folder_path = os.path.join(src_img_directory, folder_name)
            # Load the corresponding GT data
            gt_file = f'Human36M_subject{subject}_SMPL_NeuralAnnot.json'
            with open(os.path.join(gt_directory, gt_file), 'r') as f:
                gt_data = json.load(f)

            # Iterate over the images in the folder
            images = sorted(os.listdir(folder_path))
            clip_count = 0
            prevTime = time.time()
            for i in range(0, len(images), 30):
                if i + 30 <= len(images):
                    images_folder = []
                    clip_folder = f"{folder_name}_clip_{clip_count:02d}"

                    # clip_folder_path = os.path.join(dest_directory, folder_name, clip_folder)

                    # os.makedirs(clip_folder_path, exist_ok=True)
                    for image in images[i:i + 30]:
                        image_path = os.path.join(folder_path, image)
                        image = Image.open(image_path)
                        images_folder.append(image)

                    # CLIP_inputs = processor(images=images_folder, return_tensors="pt")
                    # outputs = model(**CLIP_inputs.to('cuda'))
                    # last_hidden_state = outputs.last_hidden_state

                    pt_name = clip_folder + '.pt'

                    new_dest_directory = ''

                    if any(subj in pt_name for subj in subject_list[:5]):
                        new_dest_directory = os.path.join(dest_directory, 'train')
                    elif any(subj in pt_name for subj in subject_list[-2:]):
                        new_dest_directory = os.path.join(dest_directory, 'validation')

                    pt_dest_path = os.path.join(new_dest_directory, 'feature_maps')

                    save_path = os.path.join(pt_dest_path, pt_name)
                    # torch.save(last_hidden_state.detach(), save_path)

                    # Extract and save corresponding GT data
                    gt_clip_data = [gt_data[action][subaction][str(i + frame)] for frame in range(30)]

                    single_beta = gt_clip_data[0]['shape']

                    processed_data = []
                    for frame_data in gt_clip_data:
                        # Concatenate 'pose', 'shape', into a single array
                        frame_array = np.concatenate([frame_data['pose'], single_beta])
                        processed_data.append(frame_array)

                    # Stack all frame arrays into a single 30x82 array
                    final_array = np.stack(processed_data)
                    clip_gt = torch.from_numpy(final_array)

                    # adding a row of zeros at the beginning
                    zeros = torch.zeros(1, 82)
                    clip_gt_extended = torch.cat((zeros, clip_gt), dim=0)
                    clip_gt_extended = clip_gt_extended.cpu()
                    clip_gt_extended_np = clip_gt_extended.numpy()

                    np_file_name = clip_folder + '.npy'
                    gt_path = os.path.join(new_dest_directory, 'gt')
                    np.save(os.path.join(gt_path, np_file_name), clip_gt_extended_np)

                    currTime = time.time()
                    elapsed_time = currTime - prevTime
                    print(clip_folder + 'time elapsed: ' + str(elapsed_time) + ' s')
                    prevTime = currTime
                    clip_count += 1

def reorg(reorg_path, dest_path):
    pt_folder_path = os.path.join(reorg_path, 'feature_maps')
    gt_folder_path = os.path.join(reorg_path, 'gt')

    train_dest_path = os.path.join(dest_path, 'train')
    train_ft_dest_path = os.path.join(train_dest_path, 'feature_maps')
    train_gt_dest_path = os.path.join(train_dest_path, 'gt')

    val_dest_path = os.path.join(dest_path, 'validation')
    val_ft_dest_path = os.path.join(val_dest_path, 'feature_maps')
    val_gt_dest_path = os.path.join(val_dest_path, 'gt')

    subject_list = ['s_01', 's_05', 's_06', 's_07', 's_08', 's_09',  's_11', ]

    for pt in os.listdir(pt_folder_path):
        if any(subj in pt for subj in subject_list[:5]):
            pt_path = os.path.join(pt_folder_path, pt)
            new_pt_path = os.path.join(train_ft_dest_path, pt)
            shutil.copy(pt_path, new_pt_path)

            gt_name = pt.replace('.pt','.npy')
            gt_path = os.path.join(gt_folder_path, gt_name)
            new_gt_path = os.path.join(train_gt_dest_path, gt_name)
            shutil.copy(gt_path, new_gt_path)
            print('Copied ' + pt)

        # Check if the string contains any of the last two subjects (for validation/testing)
        elif any(subj in pt for subj in subject_list[-2:]):
            pt_path = os.path.join(pt_folder_path, pt)
            new_pt_path = os.path.join(val_ft_dest_path, pt)
            shutil.copy(pt_path, new_pt_path)

            gt_name = pt.replace('.pt', '.npy')
            gt_path = os.path.join(gt_folder_path, gt_name)
            new_gt_path = os.path.join(val_gt_dest_path, gt_name)
            shutil.copy(gt_path, new_gt_path)
            print('Copied ' + pt)


if __name__ == '__main__':

    src_directory = '/media/hongji/Expansion//Human3.6M/images'
    dest_directory = '/home/hongji/Documents/h36m_data_no_sw'
    gt_directory = '/media/hongji/Expansion//Human3.6M/annotations_smpl'
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to('cuda')

    # reorg_path = '/home/hongji/Documents/data/train'
    # dest_path = '/home/hongji/Documents/h36m_data'
    # reorg(reorg_path, dest_path)


    cameras = ['ca_01', 'ca_02', 'ca_03', 'ca_04']
    for cam in cameras:
        process_images_and_gt(src_directory, dest_directory, gt_directory, cam, model)
