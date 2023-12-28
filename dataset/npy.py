import os
from PIL import Image
import json
import shutil
import numpy as np

npy_path = '/media/hongji/4T/Downloads/H36M/output_feat/s_01_act_02_subact_01_ca_01/s_01_act_02_subact_01_ca_01_clip_00/subject1_act2_subact1_clip00.npy'
    # '/media/hongji/4T/Downloads/H36M_annot_smpl/Human36M_subject1_SMPLX_NeuralAnnot.json'
# Load the numpy array from the npy file
data = np.load(npy_path, allow_pickle=True)

# If you want to print or work with the data, you can do it here
print(data)