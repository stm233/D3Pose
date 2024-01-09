import os

import torch
from PIL import Image
import json
import shutil
import numpy as np
import pickle


def process(images_path, gt_path, output_path):

    for folder_name in os.listdir(gt_path):

        full_folder_path = os.path.join(gt_path, folder_name)
        new_folder_path = os.path.join(output_path, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        for gt in os.listdir(full_folder_path):
            gt_file_path = os.path.join(full_folder_path, gt)
            with open(gt_file_path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')

            betas = data['betas'][0]

            poses = data['poses'][0]

            betas = np.array(betas)
            poses = np.array(poses)

            betas = betas[:10]

            betas = np.tile(betas, (poses.shape[0], 1))

            smpl_params = np.concatenate((poses, betas), axis=1)

            img_folder = gt.replace('.pkl', '')
            img_folder_path = os.path.join(images_path, img_folder)
            output_vid_folder = os.path.join(output_path, folder_name)
            output_vid_folder = os.path.join(output_vid_folder, img_folder)
            os.makedirs(output_vid_folder, exist_ok=True)

            images = sorted(os.listdir(img_folder_path))

            clip_count = 0

            for i in range(0, len(images), 1):
                if i + 30 < len(images):
                    clip_folder = f"_clip_{clip_count:02d}"
                    print("saving clip", clip_folder)
                    clip_folder_path = os.path.join(output_vid_folder, clip_folder)
                    os.makedirs(clip_folder_path, exist_ok=True)
                    # for img in images[i:i + 30]:
                    #     shutil.copy2(os.path.join(img_folder_path, img), clip_folder_path)

                    # adding a row of zeros at the beginning
                    clip_gt = smpl_params[i:i + 30]
                    clip_gt = torch.from_numpy(clip_gt)
                    zeros = torch.zeros(1, 82)
                    clip_gt_extended = torch.cat((zeros, clip_gt), dim=0)
                    clip_gt_extended = clip_gt_extended.cpu()
                    clip_gt_extended_np = clip_gt_extended.numpy()

                    np_file_name = f'clip{clip_count:02d}.npy'
                    np.save(os.path.join(clip_folder_path, np_file_name), clip_gt_extended_np)

                    clip_count += 1


if __name__ == '__main__':
    images_path = '/home/hongji/Documents/3DPW/images'
    gt_path = '/home/hongji/Documents/3DPW/gt'
    output_path = '/media/hongji/Expansion/3DPW/processed_data'

    process(images_path, gt_path, output_path)
