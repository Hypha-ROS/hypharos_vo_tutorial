# HyphaROS VO Tutorial

## Abstract  
A simple tutorial for monocular visual odometry implementations.  
* python3 with opencv3  
* Low dependencies (only numpy, cv2, sys and matplotlib)  
* 2D-2D VO pipeline (KLT tracker)  
* 3D-2D Localization (PnP RANSAC)  

## Environment Setup  
Recommend to setup a new virtualenv first.  
* `$ mkvirtualenv hypharos-vo --python=python3`  
* `$ workon hypharos-vo`  
* `$ cd WORKSPACE_PATH/hypharos_vo_tutorial`  
* `$ pip install -r requirement.txt`  

## Download Kitti Dataset
The example in this package uses the 00 sequence of Kitti Odometry Dataset  

## Operation
Remember to active 'hypharos-vo' virtualenv first:  
`$ workon hypharos-vo`  

### Visualize all image
`$ python visualize_all_images.py -inputs_dir ../../dataset/kitti/00/image_0/ -image_endswith png`  

### Monocular VO (2D-2D KLT)
* Command argument help function:  
`python mono_vo_kitti.py -h`  

* For virtualbox installed version (default path to dataset):  
Simple test: `$ python mono_vo_kitti.py`  
Verbose Mode: `$ python mono_vo_kitti.py --v`  
Use GPS scale info: `$ python mono_vo_kitti.py --a`  

* For specific dataset path:  
`$ python mono_vo_kitti.py -pose_path ../../dataset/kitti/poses/00.txt -image_dir ../../dataset/kitti/00/image_0/ -image_end png`  

* To stop during frame processing loop, press 'ESC' on image  

### Monocular Localization (3D-2D PnP-RANSAC)  
* Go to 'pnp' folder and execute the script:  
`$ python pnp_localization.py`  

## Developer  
* HaoChih, Lin (hypharos@gmail.com)  

## License   
Apache 2.0  


