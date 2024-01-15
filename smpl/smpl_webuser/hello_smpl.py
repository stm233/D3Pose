'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]


About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the SMPL model. The code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python hello_smpl.py

# numpy == 1.23.1

'''
import os

from serialization import load_model
import numpy as np

## Load SMPL model (here we load the female model)
## Make sure path is correct
from serialization import load_model
import numpy as np
import os

m = load_model('../models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
gt_path = '/home/hongji/Documents/data/train/gt/courtyard_arguing_00_clip_00.npy'
output_path = 'hello_world'

data = np.load(gt_path)
frame_cnt = 0

for i, row in enumerate(data):
    m.pose[:] = row[:72]
    m.betas[:] = row[-10:]
    output_obj_name = f'frame_{frame_cnt}.obj'
    output_obj_path = os.path.join(output_path, output_obj_name)

    with open(output_obj_path, 'w') as fp:
        for v in m.r:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in m.f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    frame_cnt += 1  # Increment frame count


# m.pose[:] = [0, 0, 0,
#              45/6.28,0, 0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0,
#                    0,0,0]
# print(m.pose[:])
#
# # m.betas[:] = np.random.rand(m.betas.size) * .03
# m.betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0]
#
# ## Write to an .obj file
# outmesh_path = './hello_smpl.obj'
# with open( outmesh_path, 'w') as fp:
#     for v in m.r:
#         fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
#
#     for f in m.f+1: # Faces are 1-based, not 0-based in obj files
#         fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
#
# ## Print message
# print('..Output mesh saved to: ', outmesh_path)
