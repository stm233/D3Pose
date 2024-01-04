import os
from PIL import Image
import json
import shutil
import numpy as np

def process_images_and_gt(src_img_directory, dest_directory, gt_directory):
    for folder_name in os.listdir(src_img_directory):
        # Extract subject, action, subaction from folder name
        parts = folder_name.split('_')
        subject, action, subaction = parts[1], parts[3], parts[5]
        subject = str(int(subject))
        action = str(int(action))  # Convert action index to integer and back to string to remove leading zero
        subaction = str(int(subaction))  # Same for subaction index

        # Check if the folder is for the first camera angle
        if 'ca_01' in folder_name:
            folder_path = os.path.join(src_img_directory, folder_name)
            # Load the corresponding GT data
            gt_file = f'Human36M_subject{subject}_SMPL_NeuralAnnot.json'
            with open(os.path.join(gt_directory, gt_file), 'r') as f:
                gt_data = json.load(f)

            # Iterate over the images in the folder
            images = sorted(os.listdir(folder_path))
            clip_count = 0
            for i in range(0, len(images), 30):
                if i + 30 <= len(images):
                    clip_folder = f"{folder_name}_clip_{clip_count:02d}"
                    print(clip_folder)
                    clip_folder_path = os.path.join(dest_directory, folder_name, clip_folder)
                    os.makedirs(clip_folder_path, exist_ok=True)
                    # Copy the 30 images to the new clip folder
                    for img in images[i:i+30]:
                        shutil.copy2(os.path.join(folder_path, img), clip_folder_path)

                    # Extract and save corresponding GT data
                    gt_clip_data = [gt_data[action][subaction][str(i + frame)] for frame in range(30)]

                    processed_data = []
                    for frame_data in gt_clip_data:
                        # Concatenate 'pose', 'shape', into a single array
                        frame_array = np.concatenate([frame_data['pose'], frame_data['shape']])
                        processed_data.append(frame_array)

                    # Stack all frame arrays into a single 30x82 array
                    final_array = np.stack(processed_data)

                    np_file_name = f'subject{subject}_act{action}_subact{subaction}_clip{clip_count:02d}.npy'
                    np.save(os.path.join(clip_folder_path, np_file_name), final_array)

                    clip_count += 1


if __name__ == '__main__':
    json_path = '/media/hongji/4T/Downloads/H36M/annotations_smpl/Human36M_subject1_SMPL_NeuralAnnot.json'
    # '/media/hongji/4T/Downloads/H36M_annot_smpl/Human36M_subject1_SMPLX_NeuralAnnot.json'
    with open(json_path, 'r') as file:
        data = json.load(file)
        print(data.keys())
        sub_dict = data['2']  # Access the dictionary under key '2'
        sub_dict_keys = sub_dict.keys()  # Get the keys of this sub-dictionary
        print(sub_dict_keys)  # Print the keys
        # Accessing the value associated with key '1' under the main key '2'
        value_under_key_1 = data['2']['1']['1']

        print("Value under key '2'->'1':", value_under_key_1)

    src_directory = '/media/hongji/4T/Downloads/H36M/images/images'
    dest_directory = '/media/hongji/4T/Downloads/H36M/output_feat'
    gt_directory = '/media/hongji/4T/Downloads/H36M/annotations_smpl'
    process_images_and_gt(src_directory, dest_directory, gt_directory)
