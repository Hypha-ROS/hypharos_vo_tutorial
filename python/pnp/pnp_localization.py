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
from visual_odometry_pnp import drawTrackedFeatures

import ipdb # For debug

# Global variablse
PRIFIX = '[MAIN]:'
MAX_ITERATION = 30
IMAGE_DIR = "../../../dataset/kitti/00/image_0/"
IMAGE_END = "png"

def main():
    # Find all images under image_dir (with image_end)
    image_dir = IMAGE_DIR
    image_end = IMAGE_END
    images_files_list = []
    for file in os.listdir(image_dir):
        if file.endswith(image_end):
            images_files_list.append(image_dir + '/' + file)
    if len(images_files_list) == 0:
        print(PRIFIX + '[Error] There is no image ends with ' + image_end + ' under image_dir!')
        sys.exit()
    images_files_list.sort()

    # Load data from txt (keypoints=Nx2, landmarks=Nx3)
    initial_keypoints = np.fliplr(np.genfromtxt('data/keypoints.txt', dtype='float32'))
    initial_landmarks = np.genfromtxt('data/p_W_landmarks.txt', dtype='float32')

    # Initial VisualOdometry Object
    camera_model = PinholeCamera(1241.0, 376.0, 718.8560, 
                                 718.8560, 607.1928, 185.2157)
    vo = VisualOdometry(camera_model)

    # Show the initial landmarks
    print(PRIFIX + "===== Initial keypoints and landmarks =====")
    plt.ion()
    figure_3d = plt.figure()
    landmarks_plot = figure_3d.add_subplot(111, projection='3d')
    landmarks_plot.scatter(initial_landmarks[:,0], initial_landmarks[:,1], initial_landmarks[:,2])
    landmarks_plot.set_xlabel('X Label')
    landmarks_plot.set_ylabel('Y Label')
    landmarks_plot.set_zlabel('Z Label')
    landmarks_plot.set_xlim(-20, 20)
    landmarks_plot.set_ylim(-20, 20)
    landmarks_plot.set_zlim(0, 40)
    landmarks_plot.view_init(elev=-33., azim=-90.)
    landmarks_plot.quiver(0,0,0,1,0,0,length=2.0,color=(1.0,0.0,0.0)) #draw the x-axis
    landmarks_plot.quiver(0,0,0,0,1,0,length=2.0,color=(0.0,1.0,0.0)) #draw the y-axis
    landmarks_plot.quiver(0,0,0,0,0,1,length=2.0,color=(0.0,0.0,1.0)) #draw the z-axis
    plt.draw()
    plt.pause(0.1)

    # Show the initial image with keypoints
    image_name = images_files_list[0]
    initial_frame = cv2.imread(image_name, 0)
    print(PRIFIX + "Wait any input to continue ...")
    show_image = drawTrackedFeatures(initial_frame, initial_frame, 
                                     initial_keypoints, initial_keypoints, 
                                     reinitial=True, color_circle=(0, 0, 255))
    cv2.imshow('Front Camera', show_image)
    cv2.waitKey()
   

    # Start the PnP localization process
    prev_frame = initial_frame
    prev_keypoints = initial_keypoints
    prev_landmarks = initial_landmarks
    for index in range(1, MAX_ITERATION):
        image_name = images_files_list[index]
        curr_frame = cv2.imread(image_name, 0)
        print(PRIFIX + "Process image Id[{}] ...".format(index))
        R_C_W, t_C_W, inliners, curr_keypoints_matched, prev_keypoints_matched, curr_landmarks_matched = \
            vo.processFrame(curr_frame, prev_frame, prev_keypoints, prev_landmarks)

        if(len(inliners) > 20):
            # convert from R_C_W to R_W_C
            R_W_C = R_C_W.transpose()
            t_W_C = -R_W_C @ t_C_W

            # draw the updated camera pose with respect to world frame
            landmarks_plot.quiver(t_W_C[0],t_W_C[1],t_W_C[2],
                                  R_W_C[0,0],R_W_C[1,0],R_W_C[2,0],
                                  length=2.0,color=(1.0,0.0,0.0)) #draw the x-axis
            landmarks_plot.quiver(t_W_C[0],t_W_C[1],t_W_C[2],
                                  R_W_C[0,1],R_W_C[1,1],R_W_C[2,1],
                                  length=2.0,color=(0.0,1.0,0.0)) #draw the y-axis
            landmarks_plot.quiver(t_W_C[0],t_W_C[1],t_W_C[2],
                                  R_W_C[0,2],R_W_C[1,2],R_W_C[2,2],
                                  length=2.0,color=(0.0,0.0,1.0)) #draw the z-axis
            plt.draw()
            plt.pause(0.1)

            # draw KLT tracking
            show_image = drawTrackedFeatures(curr_frame, prev_frame, 
                                             curr_keypoints_matched, prev_keypoints_matched)
            cv2.imshow('Front Camera', show_image)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

            # update the previous data
            prev_frame = curr_frame
            prev_keypoints = curr_keypoints_matched
            prev_landmarks = curr_landmarks_matched

        else:
            print(PRIFIX + "[Warning] Fail to localize!")

    #ipdb.set_trace() # for debug
    print('\n')
    input(PRIFIX + "Press [ENTER] key to exit.")
    cv2.destroyAllWindows()
    print(PRIFIX + "===== Finished ! =====")

if __name__ == '__main__':
    main()
