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
from visual_odometry import PinholeCamera, VisualOdometry
from visual_odometry import drawTrackedFeatures, getGroundTruthAndScale

#import ipdb # For debug

# Global variablse
PRIFIX = '[MAIN]:'
POSE_PATH = "../../dataset/kitti/poses/00.txt"
IMAGE_DIR = "../../dataset/kitti/00/image_0/"
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
    second_keypoints, first_keypoints = vo.bootstrapping(first_frame, second_frame)
    print(PRIFIX + "First frame id: ", str(first_index) )
    print(PRIFIX + "Second frame id: ", str(second_index) )
    print(PRIFIX + "Initial matches: ", second_keypoints.shape[0] )
    
    # Draw features tracking
    show_image = drawTrackedFeatures(second_frame, first_frame, 
                                     second_keypoints, first_keypoints)
    print(PRIFIX + "Wait any input to continue ...")
    cv2.imshow('Front Camera', show_image)
    cv2.waitKey()

    # =======================
    # ==== Frame Process ====
    # =======================
    print('\n')
    print(PRIFIX + "===== Start Frame Processing =====")
    print(PRIFIX + "Press 'ESC' to stop the loop.")
    prev_frame = second_frame
    prev_keypoints = second_keypoints
    # Main loop for frame processing
    for index in range(second_index+1, len(images_files_list)):
        curr_frame = cv2.imread(images_files_list[index], 0)
        # Get GPS ground truth trajectory
        true_pose, true_scale = getGroundTruthAndScale(poses_context, index)
        true_x, true_y = int(true_pose[0])+290, int(true_pose[2])+90        
        # Process frame w/o true scale info        
        if not absolute:
            curr_keypoints, prev_keypoints, reinitial = vo.processFrame(curr_frame, 
                                                                        prev_frame, 
                                                                        prev_keypoints)
        # Process frame w/ true scale info                                                                 
        else:
            curr_keypoints, prev_keypoints, reinitial = vo.processFrame(curr_frame, 
                                                                        prev_frame, 
                                                                        prev_keypoints,
                                                                        absolute_scale = true_scale)
        # Get VO translation                                                    
        curr_t = vo.curr_t
        if(index > 2):
            x, y, z = curr_t[0], curr_t[1], curr_t[2]
        else:
            x, y, z = 0., 0., 0.
        odom_x, odom_y = int(x)+290, int(z)+90
        if verbose:
            print('\n')
            print(PRIFIX + "Image index: ", str(index))
            print(PRIFIX + "Odometry T:\n", vo.curr_t)
            print(PRIFIX + "Odometry R:\n", vo.curr_R)
        # Draw trajectory
        cv2.circle(trajectory_plot, (odom_x,odom_y), 1, (index*255/4540,255-index*255/4540,0), 1)
        cv2.circle(trajectory_plot, (true_x,true_y), 1, (0,0,255), 2)
        cv2.rectangle(trajectory_plot, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        cv2.putText(trajectory_plot, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
        cv2.imshow('Trajectory', trajectory_plot)
        # Draw features tracking
        show_image = drawTrackedFeatures(curr_frame, prev_frame, 
                                         curr_keypoints, prev_keypoints,
                                         reinitial=reinitial)
        cv2.imshow('Front Camera', show_image)                                    
        if verbose and reinitial:
            print(PRIFIX + "Re-initial, tracked features: ", str(prev_keypoints.shape[0]))                                            
        # Handle the exit key (ESC)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        # Update the old datas
        prev_frame = curr_frame
        prev_keypoints = curr_keypoints

    #ipdb.set_trace() # for debug
    print('\n')
    input(PRIFIX + "Press any key to exit.")
    cv2.destroyAllWindows()
    print(PRIFIX + "===== Finished ! =====")

if __name__ == '__main__':
    main()
