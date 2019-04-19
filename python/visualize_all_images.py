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

# Prefix for print function
PRIFIX = '[MAIN]:'

def main():
    # Parse commands arguments
    argument = argparse.ArgumentParser()
    argument.add_argument("-inputs_dir", help="specify the directory of input images (undistorted)", required=True)
    argument.add_argument("-image_endswith", help="specify the filename extension of input images (e.g. png)", required=True)
    argument.add_argument("--v", help="set verbose as true", action="store_true")
    args = argument.parse_args()
    verbose = args.v
    # Check validation of arguments
    if not os.path.isdir(args.inputs_dir):
        print(PRIFIX + '[Error] The inputs_dir is not exist!')
        sys.exit()
    inputs_dir = os.path.realpath(args.inputs_dir)
    
    # Find all images under inputs_dir (with image_endswith)
    images_files_list = []
    for file in os.listdir(inputs_dir):
        if file.endswith(args.image_endswith):
            images_files_list.append(inputs_dir + '/' + file)
    if len(images_files_list) == 0:
        print(PRIFIX + '[Error] There is no image ends with ' + args.image_endswith + ' under inputs_dir!')
        sys.exit()
    images_files_list.sort()

    # Show basic information
    print(PRIFIX + "===== HyphaROS Mono-VO Example =====")
    print(PRIFIX + "The inputs_dir: " + inputs_dir)
    print(PRIFIX + "The image_endswith: " + args.image_endswith)
    print(PRIFIX + "Total images found: " + str(len(images_files_list)))
    print('\n')

    # Main loop for Mono-VO processing
    for file_path in images_files_list:
        image = cv2.imread(file_path, 0)
        cv2.imshow('Road facing camera', image)
        cv2.waitKey(1)

    print(PRIFIX + "===== Finished ! =====")

if __name__ == '__main__':
    main()
