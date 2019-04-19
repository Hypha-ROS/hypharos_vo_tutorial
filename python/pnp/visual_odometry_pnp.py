import numpy as np 
import cv2

import ipdb # For debug

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
        self.K = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0, 0.0, 1.0]])

# Major functions for VO computation  
class VisualOdometry:
    def __init__(self, pinhole_model):
        self.frame_stage = 0
        self.camera = pinhole_model
        self.focal = self.camera.fx
        self.center = (self.camera.cx, self.camera.cy)
        self.K = self.camera.K
        self.detector = cv2.FastFeatureDetector_create(threshold=25, 
                                                       nonmaxSuppression=True)

    # Inputs
    # @curr_frame: current cv.Mat image (greyscale)
    # @prev_frame: previous cv.Mat image (greyscale)
    # @prev_keypoints: previous 2D keypoints (Nx2)
    # Outputs
    # curr_keypoints[matches == 1]: curretn Nx2 keep tracked keypoints
    # prev_keypoints[matches == 1]: previous Nx2 keep tracked keypoints
    # matches: Nx1
    def featureTracking(self, curr_frame, prev_frame, prev_keypoints):
        # Set parameters for KLT (shape: [k,2] [k,1] [k,1])
        klt_params = dict(winSize  = (15, 15), 
                          maxLevel = 3, 
                          criteria = (cv2.TERM_CRITERIA_EPS | 
                                      cv2.TERM_CRITERIA_COUNT, 30, 0.1))
        curr_keypoints, matches, _ = cv2.calcOpticalFlowPyrLK(prev_frame, 
                                                              curr_frame, 
                                                              prev_keypoints, 
                                                              None, 
                                                              **klt_params)  
        # Remove nono-matched keypoints
        matches = matches.reshape(matches.shape[0])
        return curr_keypoints[matches == 1], prev_keypoints[matches == 1], matches
        
    def bootstrapping(self, first_frame, sceond_frame):        
        first_keypoints = self.detector.detect(first_frame)
        first_keypoints = np.array([x.pt for x in first_keypoints], dtype=np.float32)
        second_keypoints_matched, first_keypoints_matched, _ = self.featureTracking(sceond_frame, first_frame, first_keypoints)
        # Compute the initial transformation (R is 3x3, t is 3x1)
        E, mask = cv2.findEssentialMat(second_keypoints_matched, first_keypoints_matched, focal=self.focal, pp=self.center, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, curr_R, curr_t, mask = cv2.recoverPose(E, second_keypoints_matched, first_keypoints_matched, focal=self.focal, pp = self.center)
        # Create projection matrix
        P1 = self.K @ np.concatenate((np.eye(3), np.zeros((3,1))), axis=1)
        P2 = self.K @ np.concatenate((curr_R, curr_t), axis=1)
        # Estimate the 3D landmarks by triangulation (outputs is homogeneous)
        landmarks_homo = cv2.triangulatePoints(P1, P2, first_keypoints_matched.transpose(), second_keypoints_matched.transpose())
        # De-homogeneous
        landmarks = landmarks_homo.transpose()[:,0:3]/(np.tile(landmarks_homo.transpose()[:,3], (3,1)).transpose())       
        #  remove landmarks that are to far away and landmarks that are behind the camera
        landmarks_dist = np.linalg.norm(landmarks, axis=1)  
        outlier_indixes = np.logical_and(landmarks_dist < 2.0*np.mean(landmarks_dist), landmarks[:,2] > 0.0)
        return second_keypoints_matched[outlier_indixes], first_keypoints_matched[outlier_indixes], landmarks[outlier_indixes], curr_R, curr_t   

    def processFrame(self, curr_frame, prev_frame, prev_keypoints, prev_landmarks):
        # Note: Current implementation is only for localization puprose
        # Check the size of input images
        assert(curr_frame.ndim==2 and curr_frame.shape[0]==self.camera.height and curr_frame.shape[1]==self.camera.width)
        assert(prev_frame.ndim==2 and prev_frame.shape[0]==self.camera.height and prev_frame.shape[1]==self.camera.width)
        # Step-1: Feature Tracking (KLT)
        curr_keypoints_matched, prev_keypoints_matched, matches = self.featureTracking(curr_frame, prev_frame, prev_keypoints)
        curr_landmarks_matched = prev_landmarks[matches == 1]
        # Step-2: Solve PnP with RANSAC
        result = cv2.solvePnPRansac(curr_landmarks_matched, curr_keypoints_matched, self.K, np.zeros((4,1)),  
                                    iterationsCount=200, reprojectionError=10, flags=cv2.SOLVEPNP_AP3P) 
        success = result[0]
        # The output is to express the world frame with respect to camera frame
        R_C_W = cv2.Rodrigues(result[1])[0] # from Rodrigues to rotation matrix
        t_C_W = result[2] 
        inliners = result[3]

        return R_C_W, t_C_W, inliners, curr_keypoints_matched, prev_keypoints_matched, curr_landmarks_matched

