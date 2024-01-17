import os
from PIL import Image
import json
import shutil
import numpy as np
import pickle  # Import pickle module

# Path to your .pkl file
pt_root_path = '/home/hongji/Documents/data/validation/gt'
pkl_file_path = '/media/hongji/Expansion/3DPW/gt/test/downtown_arguing_00.pkl'

# for gt in os.listdir(pt_root_path):
#     # gt_path = os.path.join(pt_root_path, gt)
#     # data = np.load(gt_path)
#     #
#     # for i, row in enumerate(data):
#     #     pose = row[:72]
#     #     pose_avg = sum(pose) / len(pose)
#     #     # print("pose: ", pose_avg * 100000000000000)
#     #     betas = row[-10:]
#     #     beta_avg = sum(betas) / len(betas)
#     #     # print("betas: ", beta_avg * 100000000000000)
#     #     print("mutiples: ", pose_avg/beta_avg)

# Open the .pkl file in binary read mode
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file, encoding='latin1')

# Now you can print or work with the data
print(data.keys())
print(data['betas'])
print(data['poses'][0])
