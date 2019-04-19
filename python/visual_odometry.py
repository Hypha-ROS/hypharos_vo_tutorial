import numpy as np 
import cv2

import ipdb #for debug

# Utils Functions
def drawTrackedFeatures(second_frame, first_frame, 
                        second_keypoints, first_keypoints, reinitial=False,
                        color_line=(0, 255, 0), color_circle=(255, 0, 0)):
    mask_bgr = np.zeros_like(cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR))
    frame_bgr = cv2.cvtColor(second_frame, cv2.COLOR_GRAY2BGR)
    for i,(second,first) in enumerate(zip(second_keypoints, first_keypoints)):
        a,b = second.ravel()
        c,d = first.ravel()
        if not reinitial:
            mask_bgr = cv2.line(mask_bgr, (a,b),(c,d), color_line, 1)
        frame_bgr = cv2.circle(frame_bgr, (a,b), 3, color_circle, 1)
    return cv2.add(frame_bgr, mask_bgr)
    
def getGroundTruthAndScale(file_context, frame_id):  #specialized for KITTI odometry dataset
    ss = file_context[frame_id-1].strip().split()
    x_prev = float(ss[3])
    y_prev = float(ss[7])
    z_prev = float(ss[11])
    ss = file_context[frame_id].strip().split()
    x = float(ss[3])
    y = float(ss[7])
    z = float(ss[11])
    scale = np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
    return [x, y, z], scale

# Pinhole Camera Model
class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]

# Major functions for VO computation  
class VisualOdometry:
    def __init__(self, pinhole_model):
        self.frame_stage = 0
        self.camera = pinhole_model
        self.focal = self.camera.fx
        self.center = (self.camera.cx, self.camera.cy)
        self.curr_R = None
        self.curr_t = None
        self.detector = cv2.FastFeatureDetector_create(threshold=25, 
                                                       nonmaxSuppression=True)

    def featureTracking(self, curr_frame, prev_frame, prev_keypoints):
        # Set parameters for KLT (shape: [k,2] [k,1] [k,1])
        klt_params = dict(winSize  = (21, 21), 
                          maxLevel = 3, 
                          criteria = (cv2.TERM_CRITERIA_EPS | 
                                      cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        curr_keypoints, matches, _ = cv2.calcOpticalFlowPyrLK(prev_frame, 
                                                              curr_frame, 
                                                              prev_keypoints, 
                                                              None, 
                                                              **klt_params)  
        # Remove nono-matched keypoints
        matches = matches.reshape(matches.shape[0])
        return curr_keypoints[matches == 1], prev_keypoints[matches == 1]
        
    def bootstrapping(self, first_frame, sceond_frame):        
        first_keypoints = self.detector.detect(first_frame)
        first_keypoints = np.array([x.pt for x in first_keypoints], dtype=np.float32)
        second_keypoints_matched, first_keypoints_matched = self.featureTracking(sceond_frame, first_frame, first_keypoints)
        # Compute the initial transformation
        E, mask = cv2.findEssentialMat(second_keypoints_matched, first_keypoints_matched, focal=self.focal, pp=self.center, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.curr_R, self.curr_t, mask = cv2.recoverPose(E, second_keypoints_matched, first_keypoints_matched, focal=self.focal, pp = self.center)
        return second_keypoints_matched, first_keypoints_matched

    def processFrame(self, curr_frame, prev_frame, prev_keypoints, absolute_scale=None):
        # Check the size of input images
        assert(curr_frame.ndim==2 and curr_frame.shape[0]==self.camera.height and curr_frame.shape[1]==self.camera.width)
        assert(prev_frame.ndim==2 and prev_frame.shape[0]==self.camera.height and prev_frame.shape[1]==self.camera.width)
        # KLT Feature Tracking
        curr_keypoints_matched, prev_keypoints_matched = self.featureTracking(curr_frame, prev_frame, prev_keypoints)
        # Find Essential Matrix (by RANSAC)
        E, _ = cv2.findEssentialMat(curr_keypoints_matched, prev_keypoints_matched, focal=self.focal, pp=self.center, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # Recover Pose (the translation t is set to 1)
        _, R, t, mask = cv2.recoverPose(E, curr_keypoints_matched, prev_keypoints_matched, focal=self.focal, pp = self.center)
        # Whether or not using external scale info
        scale = 1.0
        if absolute_scale is not None:
            scale = absolute_scale        
        inliners = len(mask[mask==255])
        # Compute odometry if inliners is large enough
        if(inliners > 20):
	        self.curr_t = self.curr_t + scale*self.curr_R.dot(t) 
	        self.curr_R = R.dot(self.curr_R)
        # Re-detect the keypoints if current tracked number is too low
        reinitial = False
        if(prev_keypoints_matched.shape[0] < 1000):
            curr_keypoints_matched = self.detector.detect(curr_frame)
            curr_keypoints_matched = np.array([x.pt for x in curr_keypoints_matched], dtype=np.float32)
            reinitial = True    
        return curr_keypoints_matched, prev_keypoints_matched, reinitial
