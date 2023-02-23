# Copied from https://github.com/genea-workshop/Speech_driven_gesture_generation_with_autoencoder/blob/GENEA_2022/data_processing/bvh2features.py

# This code was written by Simon Alexanderson
# and is released here: https://github.com/simonalexanderson/PyMO

import joblib as jl
from pymo_local.writers import *
from pymo_local.viz_tools import *
from pymo_local.preprocessing import *
from pymo_local.data import Joint, MocapData
from pymo_local.parsers import BVHParser
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from argparse import ArgumentParser

import glob
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# 18 joints (only upper body)
genea_upper_body = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm',
                    'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 50 joints (upper body with fingers)
genea_upper_body_and_fingers = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1',
                                'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']

# 24 joints (upper and lower body excluding fingers)
genea_full_body = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm',
                   'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 56 joints (upper and lower body including fingers)
genea_full_body_and_fingers = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2',
                               'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']

trinity_upper_body = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm',
                      'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']


def extract_joint_angles(bvh_dir, files, dest_dir, pipeline_dir, target_joints, fps=30):
    p = BVHParser()

    data_all = list()
    for f in files:
        print(f)
        data_all.append(p.parse(f))

    print(data_all)

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
        ('root', RootNormalizer()),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        #('mir', Mirror(axis='X', append=True)),
        ('exp', MocapParameterizer('expmap')),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)

    print(out_data)
    # the datapipe will append the mirrored files to the end
    assert len(out_data) == len(files)

    jl.dump(data_pipe, os.path.join(pipeline_dir + 'data_pipe.sav'))

    fi = 0
    for f in files:
        print(f)
        np.savez(f[:-4] + ".npz", clips=out_data[fi])
        #np.savez(ff[:-4] + "_mirrored.npz", clips=out_data[len(files)+fi])
        fi = fi+1


if __name__ == '__main__':

    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--bvh_dir', '-orig', required=True,
                        help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', required=True,
                        help="Path where extracted motion features will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="./utils/",
                        help="Path where the motion data processing pipeline will be stored")
    parser.add_argument('--dataset_name', '-dataset', default="GENEA_2022",
                        help="Path where the motion data processing pipeline will be stored")

    params = parser.parse_args()

    if params.dataset_name == 'GENEA_2022':
        target_joints = genea_upper_body_and_fingers
    elif params.dataset_name == 'trinity':
        target_joints = trinity_upper_body
    else:
        raise Exception("Dataset not supported!")

    files = []
    # Go over all BVH files
    print("Going to pre-process the following motion files:")
    files = sorted([f for f in glob.iglob(params.bvh_dir+'/*.bvh')])

    extract_joint_angles(params.bvh_dir, files,
                         params.dest_dir, params.pipeline_dir, target_joints, fps=30)
