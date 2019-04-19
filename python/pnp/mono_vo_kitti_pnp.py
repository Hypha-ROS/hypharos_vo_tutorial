# ===================================================
# Copyright 2019 HyphaROS Workshop.
# Developer: HaoChih, LIN (hypha.ros@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================
import os
import sys
import cv2
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from visual_odometry_pnp import PinholeCamera, VisualOdometry
from visual_odometry_pnp import drawTrackedFeatures, getGroundTruthAndScale

import ipdb # For debug

# Global variablse
PRIFIX = '[MAIN]:'
POSE_PATH = "../../../dataset/kitti/poses/00.txt"
IMAGE_DIR = "../../../dataset/kitti/00/image_0/"
IMAGE_END = "png"

def main():
    # Parse commands arguments
    argument = argparse.ArgumentParser()
    argument.add_argument("-pose_path", help="specify the path of ground truth poses")
    argument.add_argument("-image_dir", help="specify the directory of input images (undistorted)")
    argument.add_argument("-image_end", help="specify the filename extension of input images (e.g. png)")
    argument.add_argument("--v", help="set verbose as true", action="store_true")
    argument.add_argument("--a", help="use gps scale info", action="store_true")
    args = argument.parse_args()
    verbose = args.v
    absolute = args.a

    # Check validation of arguments
    image_dir = os.path.realpath(IMAGE_DIR)
    if args.image_dir is not None:
        if os.path.isdir(args.image_dir):
            image_dir = os.path.realpath(args.image_dir)        

    pose_path = os.path.realpath(POSE_PATH)
    if args.pose_path is not None:     
        if os.path.isfile(args.pose_path):
            pose_path = os.path.realpath(args.pose_path)
    with open(pose_path) as f:
        poses_context = f.readlines()        

    image_end = IMAGE_END
    if args.pose_path is not None:
        image_end = args.image_end

    if absolute:
        if not os.path.isfile(pose_path):
            print(PRIFIX + '[Error] The ground truth poses not found, required by absolute scale info!')
            sys.exit()
    
    # Find all images under image_dir (with image_end)
    images_files_list = []
    for file in os.listdir(image_dir):
        if file.endswith(image_end):
            images_files_list.append(image_dir + '/' + file)
    if len(images_files_list) == 0:
        print(PRIFIX + '[Error] There is no image ends with ' + image_end + ' under image_dir!')
        sys.exit()
    images_files_list.sort()

    # Show basic information
    print(PRIFIX + "===== HyphaROS Mono-VO Example =====")
    print(PRIFIX + "the pose_path: " + pose_path)
    print(PRIFIX + "The image_dir: " + image_dir)
    print(PRIFIX + "The image_end: " + image_end)
    print(PRIFIX + "Use GPS scale: " + str(absolute))
    print(PRIFIX + "Total images found: " + str(len(images_files_list)))
    print('\n')

    # Initial VisualOdometry Object
    camera_model = PinholeCamera(1241.0, 376.0, 718.8560, 
                                 718.8560, 607.1928, 185.2157)
    vo = VisualOdometry(camera_model)
    trajectory_plot = np.zeros((1000,1000,3), dtype=np.uint8)

    # =======================
    # ==== Bootstrapping ====
    # =======================
    print(PRIFIX + "===== Start Bootstrapping =====")
    first_index = 0
    second_index = first_index + 3
    first_frame = cv2.imread(images_files_list[first_index], 0)
    second_frame = cv2.imread(images_files_list[second_index], 0)
    second_keypoints, first_keypoints, landmarks = vo.bootstrapping(first_frame, second_frame)
    print(PRIFIX + "First frame id: ", str(first_index) )
    print(PRIFIX + "Second frame id: ", str(second_index) )
    print(PRIFIX + "Initial matches: ", second_keypoints.shape[0] )
    
    # Draw features tracking
    show_image = drawTrackedFeatures(second_frame, first_frame, 
                                     second_keypoints, first_keypoints)
    print(PRIFIX + "Wait any input to continue ...")
    cv2.imshow('Front Camera', show_image)
    cv2.waitKey()
    
    # Draw landmarks
    figure_3d = plt.figure()
    landmarks_plot = figure_3d.add_subplot(111, projection='3d')
    landmarks_plot.scatter(landmarks[:,0], landmarks[:,1], landmarks[:,2])
    landmarks_plot.set_xlabel('X Label')
    landmarks_plot.set_ylabel('Y Label')
    landmarks_plot.set_zlabel('Z Label')
    plt.draw()
    plt.pause(0.1)

    # =======================
    # ==== Frame Process ====
    # =======================
    # TODO Implementation

    #ipdb.set_trace() # for debug
    print('\n')
    input(PRIFIX + "Press any key to exit.")
    cv2.destroyAllWindows()
    print(PRIFIX + "===== Finished ! =====")

if __name__ == '__main__':
    main()
