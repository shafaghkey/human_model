from .Skeleton import Skeleton,ConstrainedSkeleton
import nimblephysics as nimble
import torch
import numpy as np
import os
import cvxpy as cp
current_path = os.path.dirname(os.path.abspath(__file__)) + "/"

class Kalman():
    def __init__(self,dt,s,q=0.5):
        self.X = np.array([[s],[0.1],[0.01]])
        self.P = np.diag((1, 1, 1))
        self.F = np.array([[1, dt, dt*dt/2], [0, 1, dt], [0, 0, 1]])
        self.Q = np.eye(self.X.shape[0])*q
        self.Y = np.array([s])
        self.H = np.array([1, 0, 0]).reshape(1,3)
        self.R = 1 # np.eye(self.Y.shape[0])*
        
    def predict(self,dt = None):
        if dt is not None:
            self.F = np.array([[1, dt, dt*dt/2], [0, 1, dt], [0, 0, 1]])
        self.X = np.dot(self.F,self.X) #+ np.dot(self.B,self.U)
        self.P = np.dot(self.F, np.dot(self.P,self.F.T)) + self.Q

    def update(self,Y,R=None,minval=-np.inf,maxval=np.inf):
        self.Y = Y
        if R is not None:
            self.R = R
        self.K = np.dot(self.P,self.H.T) / ( self.R + np.dot(self.H,np.dot(self.P,self.H.T)) ) 
        self.X = self.X + self.K * ( Y - np.dot(self.H,self.X))
        self.P = np.dot((np.eye(self.X.shape[0])- np.dot(self.K,self.H)),self.P)
        self.X[0] = max(min(self.X[0],maxval),minval)
        self.Y = float(np.dot(self.H,self.X))
        return self.get_output()

    def get_output(self):
        return float(np.dot(self.H,self.X))

class DynamicSkeleton(ConstrainedSkeleton):
    def __init__(self, config=current_path+'BODY15_constrained_3D.xml', name=None, osim_file=None, geometry_dir='', max_velocity=None):
        
        super().__init__(config, name)
        self.timestamp = 0
        self.last_timestamp = 0
        self.keypoints_dict = {obj.name: obj for obj in self.keypoints_list}

        if osim_file is not None:
            # rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
            rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(osim_file,geometry_dir)
            self.type = 'BSM'
        else:
            rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(current_path+"bsm.osim")
            self.type = 'BSM'
        #     # print(osim_file)
        #     # rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(osim_file)
        #     rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
        #     self.type = 'rajagopal'
            # rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim('bsm.osim')
        self._nimble: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton
        
        self.measurements = []
        
        RASI = np.array([0,0.005,0.13])
        LASI = np.array([0,0.005,-0.13])
        LPSI = np.array([-0.14,0.015,-0.07])
        RPSI = np.array([-0.14,0.015,+0.07])
        
        RCAJ = np.array([0.015,-0.035,-0.02])
        RHGT = np.array([-0.05,0,0])
        LCAJ = np.array([0.015,-0.035,0.02])
        LHGT = np.array([-0.05,0,0])
        
        self.RShoulder = (RCAJ+RHGT)/2
        self.LShoulder = (LCAJ+LHGT)/2
        self.RHip = (RASI+RPSI)/2
        self.LHip = (LASI+LPSI)/2        
        
        self.s12_base = Skeleton(current_path+'BODY12.xml')
        self.skeleton_from_nimble = ConstrainedSkeleton(current_path+'BODY15_constrained_3D.xml')
        
        if  self.type == 'rajagopal':
            self.kps =  ['RKnee', 'LWrist', 'RHip', 'RShoulder',  'LElbow', 'LHip', 'RElbow', 'RWrist', 'LKnee', 'LShoulder', 'RAnkle', 'LAnkle']
            nimble_joint_names = [ 'walker_knee_r', 'radius_hand_l', 'hip_r', 'acromial_r', 'elbow_l', 'hip_l', 'elbow_r', 'radius_hand_r',  \
                        'walker_knee_l', 'acromial_l', 'ankle_r', 'ankle_l']
            self.body_dict = {'pelvis' : 'LPelvis',#LPelvis
                        'femur_r' : 'RFemur',
                        'tibia_r' : 'RTibia',
                        'talus_r' : '',
                        'calcn_r' : '',
                        'toes_r' : '',
                        'femur_l' : 'LFemur',
                        'tibia_l' : 'LTibia',
                        'talus_l' : '',
                        'calcn_l' : '',
                        'toes_l' : '',
                        'torso' : 'LClavicle',
                        'humerus_r' : 'RHumerus',
                        'ulna_r' : 'RHumerus',
                        'radius_r' : 'RForearm',
                        'hand_r' : '',
                        'humerus_l' : 'LHumerus',
                        'ulna_l' : 'LHumerus',
                        'radius_l' : 'LForearm',
                        'hand_l' : ''}
        elif self.type == "BSM":
            self.kps =  ['RKnee', 'LWrist', 'RHip', 'RShoulder',  'LElbow', 'LHip', 'RElbow', 'RWrist', 'LKnee', 'LShoulder', 'RAnkle', 'LAnkle']
            nimble_joint_names = [ 'walker_knee_r', 'wrist_l', 'hip_r', 'GlenoHumeral_r', 'elbow_l', 'hip_l', 'elbow_r', 'wrist_r',  \
                        'walker_knee_l', 'GlenoHumeral_l', 'ankle_r', 'ankle_l']
            self.body_dict = {  'pelvis':'Core', #LPelvis
                                'femur_r':'RFemur',
                                'tibia_r':'RTibia',
                                'talus_r':'',
                                'calcn_r':'',
                                'toes_r':'',
                                'femur_l':'LFemur',
                                'tibia_l':'LTibia',
                                'talus_l':'',
                                'calcn_l':'',
                                'toes_l':'',
                                'lumbar_body':'Core',#LClavicle
                                'thorax':'Core',#LClavicle
                                'head':'',
                                'scapula_r':'Core',#LClavicle
                                'humerus_r':'RHumerus',
                                'ulna_r':'RForearm',
                                'radius_r':'RForearm',
                                'hand_r':'',
                                'scapula_l':'Core',#LClavicle
                                'humerus_l':'LHumerus',
                                'ulna_l':'LForearm',
                                'radius_l':'LForearm',
                                'hand_l':''}

        self.q_l = np.ones((49))*(-180)
        self.q_u = np.ones((49))*180
        
        # Original COMETH bounds

        self.q_l[0:6], self.q_u[0:6] = -np.inf, np.inf  # pelvis
        self.q_l[6], self.q_u[6] = -40, 140     # hip_r_flexion_r
        self.q_l[7], self.q_u[7] = -45, 45      # hip_r_adduction_r
        self.q_l[8], self.q_u[8] = -45, 45      # hip_r_rotation_r
        self.q_l[13], self.q_u[13] = -40, 140   # hip_l_flexion_l
        self.q_l[14], self.q_u[14] = -45, 45    # hip_l_adduction_l
        self.q_l[15], self.q_u[15] = -45, 45    # hip_l_rotation_l
        self.q_l[9], self.q_u[9] = -10, 140     # knee_r_flexion
        self.q_l[16], self.q_u[16] = -10, 140   # knee_l_flexion
        self.q_l[10], self.q_u[10] = -20, 55    # ankle_r_flexion
        self.q_l[17], self.q_u[17] = -20, 55    # ankle_l_flexion
    
        self.q_l[20], self.q_u[20] = -20, 20    # lumbar_bending
        self.q_l[23], self.q_u[23] = -20, 20    # thorax_bending
        self.q_l[21], self.q_u[21] = -5, 5      # lumbar_extension
        self.q_l[24], self.q_u[24] = -5, 5      # thorax_extension
        self.q_l[22], self.q_u[22] = -5, 5      # lumbar_twist
        self.q_l[25], self.q_u[25] = -5, 5      # thorax_twist

        self.q_l[29], self.q_u[29] = -90, -55   # scapula_r_abduction
        self.q_l[30], self.q_u[30] = -8, 2      # scapula_r_elevation
        self.q_l[31], self.q_u[31] = -10, 40    # scapula_r_upward_rot
        self.q_l[32], self.q_u[32] = -150, 0    # shoulder_r_adduction
        self.q_l[33], self.q_u[33] = -70, 90    # shoulder_r_rotation
        self.q_l[34], self.q_u[34] = -60, 180   # shoulder_r_flexion
        self.q_l[35], self.q_u[35] = -6, 154    # elbow_r_flexion
        self.q_l[36:38], self.q_u[36:38] = -90, 90   # wrist

        self.q_l[39], self.q_u[39] = -90, -55   # scapula_l_abduction
        self.q_l[40], self.q_u[40] = -8, 2      # scapula_l_elevation
        self.q_l[41], self.q_u[41] = -10, 40    # scapula_l_upward_rot
        self.q_l[42], self.q_u[42] = 0, 150     # shoulder_l_adduction
        self.q_l[43], self.q_u[43] = -90, 70    # shoulder_l_rotation
        self.q_l[44], self.q_u[44] = -60, 180   # shoulder_l_flexion
        self.q_l[45], self.q_u[45] = -6, 154    # elbow_l_flexion
        self.q_l[46:48], self.q_u[46:48] = -90, 90   # wrist

        # # # rewrite for better comparison with names
        # self.q_l[29], self.q_u[29] = -90, 45   # scapula_r_abduction   -90, -55
        # self.q_l[30], self.q_u[30] = -8, 2      # scapula_r_elevation   -8, 2
        # self.q_l[31], self.q_u[31] = -45, 45    # scapula_r_upward_rot  -10, 40
        # self.q_l[32], self.q_u[32] = -150, 150    # shoulder_r_adduction  -150, 0
        # self.q_l[33], self.q_u[33] = -90, 90    # shoulder_r_rotation   -70, 90
        # self.q_l[34], self.q_u[34] = -150, 150   # shoulder_r_flexion    -60, 180
        self.q_l[32], self.q_u[32] = -70, 150    # shoulder_r_flexion  
        self.q_l[33], self.q_u[33] = -150, 30    # shoulder_r_adduction   
        self.q_l[34], self.q_u[34] = -90, 90   # shoulder_r_rotation    
        self.q_l[35], self.q_u[35] = -10, 154    # elbow_r_flexion       -6, 154
        # self.q_l[36:38], self.q_u[36:38] = -90, 90   # wrist  -90, 90

        # self.q_l[39], self.q_u[39] = -90, 45   # scapula_l_abduction   -90, -55
        # self.q_l[40], self.q_u[40] = -8, 2      # scapula_l_elevation   -8, 2
        # self.q_l[41], self.q_u[41] = -45, 45    # scapula_l_upward_rot  -10, 40
        # self.q_l[42], self.q_u[42] = -100, 160     # shoulder_l_adduction  0, 150
        # self.q_l[43], self.q_u[43] = -90, 90    # shoulder_l_rotation   -90, 70
        # self.q_l[44], self.q_u[44] = -160, 180   # shoulder_l_flexion    -60, 180
        # self.q_l[45], self.q_u[45] = -10, 175    # elbow_l_flexion       -6, 154
        # self.q_l[46:48], self.q_u[46:48] = -90, 90   # wrist  -90, 90
                  
        # # Add some margin to the joint limits
        # for i in range(20,26):
        #     self.q_l[i] = self.q_l[i] - 5
        #     self.q_u[i] = self.q_u[i] + 5
        #     # if self.q_l[i] != -np.inf and self.q_l[i]>-180:
        #     #     self.q_l[i] = self.q_l[i] - 20
        #     # if self.q_u[i] != np.inf and self.q_u[i]<180:
        #     #     self.q_u[i] = self.q_u[i] + 20
        
        self.q_l = self.q_l*np.pi/180
        self.q_u = self.q_u*np.pi/180

        
        self.neutral_position = self._nimble.getPositions()
        s_avg = (self.q_l + self.q_u) / 2
        self.neutral_position[6:] = s_avg[6:]
        
        if max_velocity is None:
            # Acquired from BSM dataset
            # self.qdot_l = np.array([-0.55,-0.43,-1.04,-0.74,-0.20,-0.30,-1.58,-0.57,-0.61,-1.97,0,0,0,-1.61,-0.56,-0.55,-1.97,0,0,0,-0.49,0,-0.31,-0.37,0,-0.29,0,0,0,0,0,0,-0.80,-0.84,-1.37,-1.34,-0.080,0,0,0,0,0,-0.80,-0.84,-1.42,-1.16,-0.070,0,0])
            # self.qdot_u = np.array([0.57,0.43,0.95,0.84,0.20,0.29,1.93,0.54,0.53,2.14,0,0,0,1.95,0.54,0.51,2.23,0,0,0,0.49,0,0.32,0.38,0,0.29,0,0,0,0,0,0,0.78,0.88,1.4,1.47,0.090,0,0,0,0,0,0.84,0.72,1.37,1.27,0.080,0,0])
            self.qdot_l = np.array([-1,-1,-1,-2,-2,-2,-1.58,-0.57,-0.61,-1.97,0,0,0,-1.61,-0.56,-0.55,-1.97,0,0,0,-0.49,0,-0.31,-0.37,0,-0.29,0,0,0,0,0,0,-0.80,-0.84,-1.37,-1.34,-0.080,0,0,0,0,0,-0.80,-0.84,-1.42,-1.16,-0.070,0,0])
            self.qdot_u = np.array([1,1,1,2,2,2,1.93,0.54,0.53,2.14,0,0,0,1.95,0.54,0.51,2.23,0,0,0,0.49,0,0.32,0.38,0,0.29,0,0,0,0,0,0,0.78,0.88,1.4,1.47,0.090,0,0,0,0,0,0.84,0.72,1.37,1.27,0.080,0,0])
            # # rewrite for better comparison with names
            # self.qdot_l[0:3], self.qdot_u[0:3] = -1, 1      # pelvis orientation
            # self.qdot_l[3:6], self.qdot_u[3:6] = -2, 2      # pelvis translation
            # self.qdot_l[6], self.qdot_u[6] = -1.58, 1.93    # hip_r_flexion_r
            # self.qdot_l[7], self.qdot_u[7] = -0.57, 0.54    # hip_r_adduction_r
            # self.qdot_l[8], self.qdot_u[8] = -0.61, 0.53    # hip_r_rotation_r
            # self.qdot_l[9], self.qdot_u[9] = -1.97, 2.14    # knee_r_flexion
            # self.qdot_l[10], self.qdot_u[10] = 0, 0         # ankle_r_flexion
            # self.qdot_l[11], self.qdot_u[11] = 0, 0         # subtalar_r
            # self.qdot_l[12], self.qdot_u[12] = 0, 0         # mtp_r
            # self.qdot_l[13], self.qdot_u[13] = -1.61, 1.95  # hip_l_flexion_l
            # self.qdot_l[14], self.qdot_u[14] = -0.56, 0.54  # hip_l_adduction_l
            # self.qdot_l[15], self.qdot_u[15] = -0.55, 0.51  # hip_l_rotation_l
            # self.qdot_l[16], self.qdot_u[16] = -1.97, 2.23  # knee_l_flexion
            # self.qdot_l[17], self.qdot_u[17] = 0, 0         # ankle_l_flexion
            # self.qdot_l[18], self.qdot_u[18] = 0, 0         # subtalar_l
            # self.qdot_l[19], self.qdot_u[19] = 0, 0         # mtp_l
            # self.qdot_l[20], self.qdot_u[20] = -0.49, 0.49  # lumbar_bending
            # self.qdot_l[21], self.qdot_u[21] = 0, 0         # lumbar_extension
            # self.qdot_l[22], self.qdot_u[22] = -0.31, 0.32  # lumbar_twist
            # self.qdot_l[23], self.qdot_u[23] = -0.37, 0.38  # thorax_bending
            # self.qdot_l[24], self.qdot_u[24] = 0, 0         # thorax_extension
            # self.qdot_l[25], self.qdot_u[25] = -0.29, 0.29  # thorax_twist
            # self.qdot_l[26:28], self.qdot_u[26:28] = 0, 0   # neck_bending, extension, twist
            # self.qdot_l[29], self.qdot_u[29] = -0.80, 0.78  # scapula_r_abduction
            # self.qdot_l[30], self.qdot_u[30] = -0.84, 0.88  # scapula_r_elevation
            # self.qdot_l[31], self.qdot_u[31] = -1.37, 1.4   # scapula_r_upward_rot
            # self.qdot_l[32], self.qdot_u[32] = -1.34, 1.47  # shoulder_r_adduction
            # self.qdot_l[33], self.qdot_u[33] = -0.08, 0.09  # shoulder_r_rotation
            # self.qdot_l[34], self.qdot_u[34] = -1, 1        # shoulder_r_flexion
            # self.qdot_l[35], self.qdot_u[35] = -0.80, 0.84  # elbow_r_flexion
            # self.qdot_l[36:38], self.qdot_u[36:38] = -1, 1  # radioulnar_r_pro_sup, wrist
            # self.qdot_l[39], self.qdot_u[39] = -0.80, 0.84  # scapula_l_abduction
            # self.qdot_l[40], self.qdot_u[40] = -0.72, 0.72  # scapula_l_elevation
            # self.qdot_l[41], self.qdot_u[41] = -1.37, 1.37  # scapula_l_upward_rot
            # self.qdot_l[42], self.qdot_u[42] = -1.27, 1.27  # shoulder_l_adduction
            # self.qdot_l[43], self.qdot_u[43] = -0.08, 0.08  # shoulder_l_rotation
            # self.qdot_l[44], self.qdot_u[44] = -1, 1        # shoulder_l_flexion
            # self.qdot_l[45], self.qdot_u[45] = -0.80, 0.84  # elbow_l_flexion
            # self.qdot_l[46:48], self.qdot_u[46:48] = -1, 1  # radioulnar_l_pro_sup, wrist

        else:
            self.qdot_l = np.zeros(self.q_u.shape)-max_velocity
            self.qdot_u = np.zeros(self.q_u.shape)+max_velocity
        
        self.prob = None
        self.prev_mask = None
        self.kf = None
        self.joints = [self._nimble.getJoint(l) for l in nimble_joint_names]
        pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
        self.s12_base.load_from_numpy(pos.reshape(-1,3),self.kps)
        self.skeleton_from_nimble.load_from_BODY12(self.s12_base)
        
        # Save for faster qpIK
        self.qpIK_problems = {}
        
        self.reset()
        
    def reset_history(self):
        for b in self.bones_list:
            b.history = []
        for kp in self.keypoints_list:
            kp._history = []
        self.height_history = []
        self.measurements = []
    
    def reset(self):
        self._nimble.setPositions(self.neutral_position)
        self.reset_history()
        self.kf = None
    
    def estimate_confidence(self):
        # Update each keypoint.confidence value
        h = np.nanmean(self.height_history)
        for b in self.bones_list:
            min_l = (self.proportions[b.name][0]-2*self.proportions[b.name][1])*self.height_history[-1]
            max_l = (self.proportions[b.name][0]+2*self.proportions[b.name][1])*self.height_history[-1]
            # If they are in range, increase confidence
            if b.length > min_l and b.length < max_l:
                    b.src.confidence  = min( b.src.confidence + 0.1, 1) 
                    b.dest.confidence  = min( b.dest.confidence + 0.1, 1) 
            else:
                    b.src.confidence  = max( b.src.confidence - 0.1, 0) 
                    b.dest.confidence  = max( b.dest.confidence - 0.1, 0)
        
    
    # Remove the joint position to place the corrected hip (closer to ASI and PSI)
    def correct(self,pos):
        # Correct the pelvis joint
        transform = self._nimble.getBodyNode('pelvis').getWorldTransform()
        scale = self._nimble.getBodyNode('pelvis').getScale()
        pos[3*self.kps.index("RHip"):3*self.kps.index("RHip")+3] = transform.multiply(np.multiply(scale,self.RHip))
        pos[3*self.kps.index("LHip"):3*self.kps.index("LHip")+3] = transform.multiply(np.multiply(scale,self.LHip))
        
        # # Correct the scapula joints
        transform = self._nimble.getBodyNode('scapula_l').getWorldTransform()
        pos[3*self.kps.index("LShoulder"):3*self.kps.index("LShoulder")+3] = transform.multiply(self.LShoulder)
        transform = self._nimble.getBodyNode('scapula_r').getWorldTransform()
        pos[3*self.kps.index("RShoulder"):3*self.kps.index("RShoulder")+3] = transform.multiply(self.RShoulder)
        return pos
    
    # Scaling better suited for noisy input (e.g., marker-less data)
    def estimate_scale(self):
        scale =  self._nimble.getBodyScales().reshape(-1,3)
        # If there may be error is the height and bones estimation, return the mean of the previous
        if np.all(np.isnan(self.height_history)):
            return
        # print("here")
        h = np.nanmean(self.height_history)
        for i,b in enumerate(self.body_dict.keys()):
            if self.body_dict[b] == 'Core' or self.body_dict[b] == '':
                scale[i,:] = h / self.skeleton_from_nimble.estimate_height()
            else:
                sc = np.nanmean(self.bones_dict[self.body_dict[b]].history) / self.skeleton_from_nimble.bones_dict[self.body_dict[b]].length
                # If there is a symmetrical one
                if self.body_dict[b] in self.symmetry:
                    if np.isnan(sc):
                        sc = np.nanmean(self.bones_dict[self.symmetry[self.body_dict[b]]].history) / self.skeleton_from_nimble.bones_dict[self.symmetry[self.body_dict[b]]].length
                    else:
                        sc_sym = np.nanmean(self.bones_dict[self.symmetry[self.body_dict[b]]].history) / self.skeleton_from_nimble.bones_dict[self.symmetry[self.body_dict[b]]].length
                        if np.abs(1-sc) > np.abs(1-sc_sym): 
                            sc = sc_sym
                            # print("symmetric law for",b)
                if not np.isnan(sc):
                    scale[i,:] = sc
        
        # Clip the scaling between fixed bounds
        # scale = np.clip(scale,0.85,1.15)
        avg_scale = np.mean(scale)
        scale = np.clip(scale,avg_scale-0.05,avg_scale+0.05)
        
        self._nimble.setBodyScales(scale.reshape(-1,1))


    # # Old scaling version, only for precise input (e.g., marker-based)
    # def scale(self):
    #     scale =  self._nimble.getBodyScales().reshape(-1,3)
    #     # If there may be error is the height and bones estimation, return the mean of the previous
    #     if np.all(np.isnan(self.height_history)):
    #         return
    #     h = np.nanmean(self.height_history)
    #     for i,b in enumerate(self.body_dict.keys()):
    #         if self.body_dict[b] == 'Core' or self.body_dict[b] == '':
    #             scale[i,:] = h / self.skeleton_from_nimble.estimate_height()
    #         else:
    #             sc = np.nanmean(self.bones_dict[self.body_dict[b]].history) / self.skeleton_from_nimble.bones_dict[self.body_dict[b]].length
    #             if np.isnan(sc) and self.body_dict[b] in self.symmetry:
    #                 sc = np.nanmean(self.bones_dict[self.symmetry[self.body_dict[b]]].history) / self.skeleton_from_nimble.bones_dict[self.symmetry[self.body_dict[b]]].length
    #             if not np.isnan(sc):
    #                 scale[i,:] = sc
    #     # print(np.round(scale[:,0].transpose(),2))
    #     self._nimble.setBodyScales(scale.reshape(-1,1))
            
    # Inverse kinematics through gradient descend
    def exact_scale(self,max_iterations=1000,precision=0.001):
        older_loss = np.inf
        mask = ~np.isnan(super().to_numpy(self.kps)) 
        target = super().to_numpy(self.kps)[mask].reshape(1,-1).squeeze()
        for _ in range(max_iterations):
            # Angular position placement
            q = self._nimble.getPositions()
            i=0
            while i < max_iterations:
                pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
                # pos = np.array(self._nimble.getJointWorldPositions(self.joints))
                pos = pos[mask.reshape(1,-1).squeeze()]
                d_loss_d__pos = 2 * (pos - target)
                d_pos_d_joint_angles = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
                d_pos_d_joint_angles = d_pos_d_joint_angles[mask.reshape(1,-1).squeeze(),:]
                d_loss_d_joint_angles = d_pos_d_joint_angles.T @ d_loss_d__pos
                q -= 0.05 * d_loss_d_joint_angles            
                q = np.clip(q,self.q_l,self.q_u)
                self._nimble.setPositions(q)
                i+=1
            
            # Scaling setting
            scale =  self._nimble.getBodyScales()
            j = 0
            while j < max_iterations:
                pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
                # pos = np.array(self._nimble.getJointWorldPositions(self.joints))
                pos = pos[mask.reshape(1,-1).squeeze()]
                d_loss_d__pos = 2 * (pos - target)
                d_pos_d_scales = self._nimble.getJointWorldPositionsJacobianWrtBodyScales(self.joints)
                d_pos_d_scales = d_pos_d_scales[mask.reshape(1,-1).squeeze(),:]
                # d_pos_d_scales = d_pos_d_scales[mask.reshape(1,-1).squeeze(),0:72:3]
                d_loss_d_scales = d_pos_d_scales.T @ d_loss_d__pos
                # d_loss_d_scales = d_loss_d_scales.reshape((1,-1))
                # d_loss_d_scales = np.array([d_loss_d_scales,d_loss_d_scales,d_loss_d_scales]).transpose()
                # d_loss_d_scales = d_loss_d_scales.squeeze().reshape((-1,))
                scale -= 0.001 * d_loss_d_scales
                self._nimble.setBodyScales(scale)
                j+=1

            error = np.array(self._nimble.getJointWorldPositions(self.joints))[mask.reshape(1,-1).squeeze()] - target
            loss = np.inner(error, error)
            if np.abs(older_loss - loss) < precision:
                # print(loss)
                break
            older_loss = loss
            





    def qpIK(self,max_iterations=100,dt=100,precision=0.00001):
        data_in = super().to_numpy(self.kps)
        mask = ~np.isnan(data_in)
        
        nkey = np.sum(mask[:,0])
        key = str(np.sum(mask[:,0]))+"."
            
        # TODO: it is important to log?
        # print(key,key in self.qpIK_problems.keys())
        
        problem_to_build = False if key in self.qpIK_problems.keys() else True
                    
        subset_joints = [self.joints[i] for i in range(len(self.joints)) if mask[i,0]]
        
        x_target = data_in[mask].reshape(1,-1).squeeze()
        
        if problem_to_build:        
            self.q = cp.Parameter((49,))
            self.x = cp.Parameter((nkey*3,))
            self.J = cp.Parameter((nkey*3,49))
            self.x_target = cp.Parameter((nkey*3,))
            self.delta = cp.Variable((nkey*3,))
            self.dq = cp.Variable((49,))
            self.constraints = [self.x + self.J@self.dq == self.x_target + self.delta]  
            # self.constraints += [-self.dq[6:] >= -1*(self.q_u[6:]-self.q[6:]), self.dq[6:] >= -1*(self.q[6:]-self.q_l[6:])]
            self.constraints += [-self.dq >= -1*(self.q_u-self.q), self.dq >= -1*(self.q-self.q_l)]
            self.dq_prev = cp.Parameter((49,))

            # Velocity constraints
            self.dq_l = cp.Parameter((49,))
            self.dq_u = cp.Parameter((49,))
            # self.constraints += [self.dq_prev[6:] + self.dq[6:] >= self.dq_l[6:], self.dq_prev[6:] + self.dq[6:] <= self.dq_u[6:]]
            self.constraints += [self.dq_prev + self.dq >= self.dq_l, self.dq_prev + self.dq <= self.dq_u]
            self.obj = cp.Minimize( cp.quad_form(self.delta,np.eye(self.delta.shape[0])) + cp.quad_form(self.dq,np.eye(self.dq.shape[0])) )
            self.prob = cp.Problem(self.obj, self.constraints)
            self.qpIK_problems[key] = {"problem": self.prob, 
                                                   "x_target":self.x_target,
                                                   "x" : self.x,
                                                   "J" : self.J,
                                                   "delta" : self.delta,
                                                   "dq_l" : self.dq_l,
                                                   "dq_u" : self.dq_u,
                                                   "dq_prev" : self.dq_prev,
                                                   "dq" : self.dq,
                                                   "q" : self.q
                                                   }
        else:
            self.prob = self.qpIK_problems[key]["problem"]
            self.x_target = self.qpIK_problems[key]["x_target"]
            self.x = self.qpIK_problems[key]["x"]
            self.J = self.qpIK_problems[key]["J"]
            self.delta = self.qpIK_problems[key]["delta"]
            self.dq_l = self.qpIK_problems[key]["dq_l"]
            self.dq_u = self.qpIK_problems[key]["dq_u"]
            self.dq_prev = self.qpIK_problems[key]["dq_prev"]
            self.dq = self.qpIK_problems[key]["dq"] 
            self.q = self.qpIK_problems[key]["q"]
        self.dq_l.value = dt*self.qdot_l
        self.dq_u.value = dt*self.qdot_u
        self.dq_prev.value = np.zeros(self.q.shape)
        self.x_target.value = x_target
        
                
        older_loss = np.inf
        i=0
        while i < max_iterations:
            self.q.value = self._nimble.getPositions()
            x = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
            J = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
            self.J.value = J[mask.reshape(1,-1).squeeze(),:]
            self.x.value = x[mask.reshape(1,-1).squeeze()]
            
            error = self.x.value - self.x_target.value
            loss = np.inner(error, error)
            if np.abs(older_loss - loss) < precision:
                # TODO: it is important to log?
                # print("RUN N",i)
                break
            older_loss = loss
            
            # self.prob.solve(solver=cp.ECOS)
            self.prob.solve(solver=cp.OSQP)
            # print(i,self.prob.status,type(self.dq.value))
            self.dq_prev.value += np.array(self.dq.value)
            self._nimble.setPositions(self.q.value+self.dq.value) # *0.01
            
            i+=1
    
    def multisource_qpIK(self,targets,max_iterations=100,dt=0.02,precision=0.00001):
        
        masks = [~np.isnan(t) for t in targets]
        
        nkey = np.sort([int(np.sum(m[:,0])) for m in masks])
        
        permutation = np.argsort([int(np.sum(m[:,0])) for m in masks])
        
        targets = [targets[i] for i in permutation]
        masks = [masks[i] for i in permutation]
        
        key = ""
        for k in nkey.tolist():
            key+=str(k)+"."
        
        # TODO: remove?
        # print(key,key in self.qpIK_problems.keys())
        
        problem_to_build = False if key in self.qpIK_problems.keys() else True
            
        subsets_joints = []
        for mask in masks:
            subsets_joints.append([self.joints[i] for i in range(len(self.joints)) if mask[i,0]])
                
        if np.all(self._nimble.getPositions() == self.neutral_position):
            dt = 100
        
        # Every time set a new problem. It's slower but can be improved
        if problem_to_build:
            self.q = cp.Parameter((49,))
            self.xs = [cp.Parameter((nk*3,)) for nk in nkey]
            self.Js = [cp.Parameter((nk*3,49)) for nk in nkey]
            self.x_targets = [cp.Parameter((nk*3,)) for nk in nkey]
            self.deltas =  [cp.Variable((nk*3,)) for nk in nkey]
            self.dq = cp.Variable((49,))
            self.constraints = []
            self.constraints += [self.xs[i] + self.Js[i]@self.dq == self.x_targets[i] + self.deltas[i] for i in range(len(masks))]
            # Joint limits
            # self.constraints += [-self.dq[6:] >= -1*(self.q_u[6:]-self.q[6:]), self.dq[6:] >= -1*(self.q[6:]-self.q_l[6:])]
            self.constraints += [-self.dq >= -1*(self.q_u-self.q), self.dq >= -1*(self.q-self.q_l)]
            self.dq_prev = cp.Parameter((49,))
            # Velocity constraints
            self.dq_l = cp.Parameter((49,))
            self.dq_u = cp.Parameter((49,))
            # self.constraints += [self.dq_prev[6:] + self.dq[6:] >= self.dq_l[6:], self.dq_prev[6:] + self.dq[6:] <= self.dq_u[6:]]
            self.constraints += [self.dq_prev + self.dq >= self.dq_l, self.dq_prev + self.dq <= self.dq_u]
            to_minimize = cp.quad_form(self.dq,np.eye(self.dq.shape[0]))
            for delta in self.deltas:
                to_minimize += cp.quad_form(delta,np.eye(delta.shape[0]))
            self.obj = cp.Minimize(to_minimize)
            self.prob = cp.Problem(self.obj, self.constraints)
            self.qpIK_problems[key] = {"problem": self.prob, 
                                                   "x_targets":self.x_targets,
                                                   "xs" : self.xs,
                                                   "Js" : self.Js,
                                                   "deltas" : self.deltas,
                                                   "dq_l" : self.dq_l,
                                                   "dq_u" : self.dq_u,
                                                   "dq_prev" : self.dq_prev,
                                                   "dq" : self.dq,
                                                   "q" : self.q
                                                   }
        else:
            self.prob = self.qpIK_problems[key]["problem"]
            self.x_targets = self.qpIK_problems[key]["x_targets"]
            self.xs = self.qpIK_problems[key]["xs"]
            self.Js = self.qpIK_problems[key]["Js"]
            self.deltas = self.qpIK_problems[key]["deltas"]
            self.dq_l = self.qpIK_problems[key]["dq_l"]
            self.dq_u = self.qpIK_problems[key]["dq_u"]
            self.dq_prev = self.qpIK_problems[key]["dq_prev"]
            self.dq = self.qpIK_problems[key]["dq"]
            self.q = self.qpIK_problems[key]["q"]
        
        # self.Rdiag.value = np.diag([self.keypoints_dict[kp].confidence for kp in self.kps for _ in range(3)])
                
        self.dq_l.value = dt*self.qdot_l
        self.dq_u.value = dt*self.qdot_u
        self.dq_prev.value = np.zeros(self.q.shape)
        for i,x_target in enumerate(self.x_targets):
            x_target.value = targets[i][masks[i]]
        
                
        older_loss = np.inf
        while i < max_iterations:
            self.q.value = self._nimble.getPositions()
            x = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
            J = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
            for j in range(len(self.x_targets)):
                self.Js[j].value = J[masks[j].reshape(1,-1).squeeze(),:]
                self.xs[j].value = x[masks[j].reshape(1,-1).squeeze()]
            
            error = np.nanmean([self.xs[j].value - self.x_targets[j].value])
            loss = np.inner(error, error)
            # print("loss",loss)
            if np.abs(older_loss - loss) < precision:
                print("MRUN N",i)
                break
            older_loss = loss
            
            # self.prob.solve(solver=cp.ECOS)
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            # print(i,self.prob.status,type(self.dq.value))
            self.dq_prev.value += np.array(self.dq.value)
            self._nimble.setPositions(self.q.value+self.dq.value) # *0.01
            i+=1
    
    def qpIK2D(self,kps2d,K,max_iterations=100000,dt=0.02,precision=0.00001):
        
        # kps2d is a list of matrices of size (|kps|,2)
        mask = ~np.isnan(kps2d)
        
        nkey = np.sum(mask[:,0])
                
        key=str(nkey)+"."
                
        problem_to_build = False if key in self.qpIK_problems.keys() else True
            
        subsets_joints = []
        
        subsets_joints.append([self.joints[i] for i in range(len(self.joints)) if mask[i,0]])
                
        if np.all(self._nimble.getPositions() == self.neutral_position):
            dt = 100
                
        # Every time set a new problem. It's slower but can be improved
        if problem_to_build:
            # Variables
            self.deltas =  cp.Variable((nkey*3,1))
            self.dq = cp.Variable((49,1))
            self.z =  cp.Variable((nkey*3,1),nonneg=True)
            # Parameters            
            self.q = cp.Parameter((49,1))
            self.xs = cp.Parameter((nkey*3,1))
            self.Js = cp.Parameter((nkey*3,49))
            self.dq_prev = cp.Parameter((49,1))
            self.dq_l = cp.Parameter((49,1))
            self.dq_u = cp.Parameter((49,1))
            self.A = cp.Parameter((nkey*3,1))
            # Constraints
            self.constraints = []
            
            for i in range(0,nkey,3):
                self.constraints += [self.z[i] == self.z[i+1], self.z[i] == self.z[i+2]]
            
            self.constraints += [self.xs + self.Js@self.dq == cp.multiply(self.A,self.z) + self.deltas]
            
            self.constraints += [z <= 10 for z in self.z]
            self.constraints += [z >= 1 for z in self.z]
            
            self.constraints += [-self.dq >= -1*(self.q_u.reshape(49,1)-self.q), self.dq >= -1*(self.q-self.q_l.reshape(49,1))]
            self.constraints += [self.dq_prev + self.dq >= self.dq_l, self.dq_prev + self.dq <= self.dq_u]
            
            # Problem
            to_minimize = cp.quad_form(self.dq,np.eye(self.dq.shape[0]))
            to_minimize += cp.quad_form(self.deltas,np.eye(self.deltas.shape[0]))
            # to_minimize += cp.quad_form(cp.multiply(-1,self.z),np.eye(self.z.shape[0]))
            # to_minimize += cp.quad_form(self.z,np.eye(self.z.shape[0]))
            
            self.obj = cp.Minimize(to_minimize)
            
            # self.obj +=
            
            self.prob = cp.Problem(self.obj, self.constraints)
            self.qpIK_problems[key] = {"problem": self.prob, 
                                        #    "x_targets":self.x_targets,
                                        "xs" : self.xs,
                                        "Js" : self.Js,
                                        "deltas" : self.deltas,
                                        "dq_l" : self.dq_l,
                                        "dq_u" : self.dq_u,
                                        "dq_prev" : self.dq_prev,
                                        "dq" : self.dq,
                                        "q" : self.q,
                                        "z" : self.z,
                                        "A" : self.A,
                                            }
        else:
            self.prob = self.qpIK_problems[key]["problem"]
            self.xs = self.qpIK_problems[key]["xs"]
            self.Js = self.qpIK_problems[key]["Js"]
            self.deltas = self.qpIK_problems[key]["deltas"]
            self.dq_l = self.qpIK_problems[key]["dq_l"]
            self.dq_u = self.qpIK_problems[key]["dq_u"]
            self.dq_prev = self.qpIK_problems[key]["dq_prev"]
            self.dq = self.qpIK_problems[key]["dq"]
            self.q = self.qpIK_problems[key]["q"]
            self.z = self.qpIK_problems[key]["z"]
            self.A = self.qpIK_problems[key]["A"]
                
        self.dq_l.value = dt*self.qdot_l.reshape(49,1)
        self.dq_u.value = dt*self.qdot_u.reshape(49,1)
        self.dq_prev.value = np.zeros(self.q.shape)

        U = np.hstack([kps2d,np.ones((kps2d.shape[0],1))]).reshape(-1,1)
        C = np.tile(np.array([K[0,2],K[1,2],0]),kps2d.shape[0]).reshape(-1,1)
        F = np.tile(np.array([K[0,0],K[1,1],1]),kps2d.shape[0]).reshape(-1,1)

        self.A.value = (U-C)/F
                
        # older_loss = np.inf
        i = 0
        while i < max_iterations:
            self.q.value = self._nimble.getPositions().reshape(49,1)
            self.xs.value = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints))).reshape(3*nkey,1)
            self.Js.value = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
                        
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            # print(i,self.prob.status,type(self.dq.value))
            # print(self.z.value)
            self.dq_prev.value += np.array(self.dq.value)
            self._nimble.setPositions(self.q.value+self.dq.value) # *0.01
            i+=1
    
    
    def qpIK2Dw(self,kps2d,K,R,T,max_iterations=100000,dt=0.02,precision=0.00001):
        
        # kps2d is a list of matrices of size (|kps|,2)
        mask = ~np.isnan(kps2d)
        
        nkey = np.sum(mask[:,0])
                
        key=str(nkey)+"."
                
        problem_to_build = False if key in self.qpIK_problems.keys() else True
            
        subsets_joints = []
        
        subsets_joints.append([self.joints[i] for i in range(len(self.joints)) if mask[i,0]])
                
        if np.all(self._nimble.getPositions() == self.neutral_position):
            dt = 100
                
        # Every time set a new problem. It's slower but can be improved
        if problem_to_build:
            # Variables
            self.deltas =  cp.Variable((nkey*3,1))
            self.dq = cp.Variable((49,1))
            self.z =  cp.Variable((nkey*3,1),nonneg=True)
            # Parameters            
            self.q = cp.Parameter((49,1))
            self.xs = cp.Parameter((nkey*3,1))
            self.Js = cp.Parameter((nkey*3,49))
            self.dq_prev = cp.Parameter((49,1))
            self.dq_l = cp.Parameter((49,1))
            self.dq_u = cp.Parameter((49,1))
            self.A = cp.Parameter((nkey*3,1))
            # self.R = cp.Parameter((nkey*3, nkey*3))
            # self.T = cp.Parameter((nkey*3,1))
            self.T = np.tile(T.reshape(1,-1),nkey).reshape(-1,1)
            # print(self.T.value)
            
            R_e = np.zeros((3*nkey, 3*nkey))
            # Place the 3x3 matrix on the diagonal
            for i in range(nkey):
                R_e[i*3:(i+1)*3, i*3:(i+1)*3] = R
            self.R = R_e
            # Constraints
            self.constraints = []
            
            for i in range(0,nkey,3):
                self.constraints += [self.z[i] == self.z[i+1], self.z[i] == self.z[i+2]]
            
            self.constraints += [self.xs + self.Js@self.dq == self.R@cp.multiply(self.A,self.z) + self.T + self.deltas]
            
            self.constraints += [z <= 10 for z in self.z]
            self.constraints += [z >= 1 for z in self.z]
            
            # self.constraints += [self.xs[i] + self.Js[i]@self.dq == self.x_targets[i] + self.deltas[i] for i in range(len(masks))]
            self.constraints += [-self.dq >= -1*(self.q_u.reshape(49,1)-self.q), self.dq >= -1*(self.q-self.q_l.reshape(49,1))]
            self.constraints += [self.dq_prev + self.dq >= self.dq_l, self.dq_prev + self.dq <= self.dq_u]
            
            # Problem
            to_minimize = cp.quad_form(self.dq,np.eye(self.dq.shape[0]))
            to_minimize += cp.quad_form(self.deltas,np.eye(self.deltas.shape[0]))
            # to_minimize += cp.quad_form(self.z,np.eye(self.z.shape[0]))
            
            self.obj = cp.Minimize(to_minimize)
            self.prob = cp.Problem(self.obj, self.constraints)
            self.qpIK_problems[key] = {"problem": self.prob, 
                                        #    "x_targets":self.x_targets,
                                        "xs" : self.xs,
                                        "Js" : self.Js,
                                        "deltas" : self.deltas,
                                        "dq_l" : self.dq_l,
                                        "dq_u" : self.dq_u,
                                        "dq_prev" : self.dq_prev,
                                        "dq" : self.dq,
                                        "q" : self.q,
                                        "z" : self.z,
                                        "A" : self.A,
                                        "R" : self.R,
                                        "T" : self.T,
                                            }
        else:
            # print(self.qpIK_problems[key].keys())
            self.prob = self.qpIK_problems[key]["problem"]
            self.xs = self.qpIK_problems[key]["xs"]
            self.Js = self.qpIK_problems[key]["Js"]
            self.deltas = self.qpIK_problems[key]["deltas"]
            self.dq_l = self.qpIK_problems[key]["dq_l"]
            self.dq_u = self.qpIK_problems[key]["dq_u"]
            self.dq_prev = self.qpIK_problems[key]["dq_prev"]
            self.dq = self.qpIK_problems[key]["dq"]
            self.q = self.qpIK_problems[key]["q"]
            self.z = self.qpIK_problems[key]["z"]
            self.A = self.qpIK_problems[key]["A"]
            self.R = self.qpIK_problems[key]["R"]
            self.T = self.qpIK_problems[key]["T"]
                
        self.dq_l.value = dt*self.qdot_l.reshape(49,1)
        self.dq_u.value = dt*self.qdot_u.reshape(49,1)
        self.dq_prev.value = np.zeros(self.q.shape)

        U = np.hstack([kps2d,np.ones((nkey,1))]).reshape(-1,1)
        C = np.tile(np.array([K[0,2],K[1,2],0]),nkey).reshape(-1,1)
        F = np.tile(np.array([K[0,0],K[1,1],1]),nkey).reshape(-1,1)

        self.A.value = (U-C)/F
                
        # older_loss = np.inf
        i = 0
        while i < max_iterations:
            self.q.value = self._nimble.getPositions().reshape(49,1)
            self.xs.value = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints))).reshape(3*nkey,1)
            # print(self.xs.value.reshape(-1,3))
            self.Js.value = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
                        
            self.prob.solve(warm_start=True,solver=cp.OSQP) #, warm_start=True,solver=cp.ECOS  verbose=True
            # print(i,self.prob.status,type(self.dq.value))
            # sol = self.R.value@(self.A.value*self.z.value) + self.T.value
            # print(sol.reshape(-1,3))
            # print("z:",self.z.value)
                        
            # if self.prob.status != "optimal":
            #     return
            
            # print(self.z.value)
            self.dq_prev.value += np.array(self.dq.value)
            self._nimble.setPositions(self.q.value+self.dq.value) # *0.01
            i+=1
    
    def multisource_qpIK2D(self,kps2ds,Ks,Rs,Ts,camera_names,max_iterations=100000,dt=0.02,precision=0.00001):
        
        # kps2d is a list of matrices of size (|kps|,2)
        masks = [~np.isnan(kps2d) for kps2d in kps2ds]
        
        nkeys = [np.sum(mask[:,0]) for mask in masks]
                
        key=".".join([str(c) for c in camera_names])+"."
                
        for k in nkeys:
            key+=str(k)+"."
        
        problem_to_build = False if key in self.qpIK_problems.keys() else True
            
        subsets_joints = []
        for mask in masks:
            subsets_joints.append([self.joints[i] for i in range(len(self.joints)) if mask[i,0]])
           
        if np.all(self._nimble.getPositions() == self.neutral_position):
            dt = 100
        
        # Every time set a new problem. It's slower but can be improved
        if problem_to_build:
            # Variables
            self.deltas =  [cp.Variable((nkey*3,1)) for nkey in nkeys]
            self.dq = cp.Variable((49,1))
            self.z =  [cp.Variable((nkey*3,1),nonneg=True) for nkey in nkeys]
            # Parameters            
            self.q = cp.Parameter((49,1))
            self.xs = [cp.Parameter((nkey*3,1)) for nkey in nkeys]
            self.Js = [cp.Parameter((nkey*3,49)) for nkey in nkeys]
            self.dq_prev = cp.Parameter((49,1))
            self.dq_l = cp.Parameter((49,1))
            self.dq_u = cp.Parameter((49,1))
            self.A = [cp.Parameter((nkey*3,1)) for nkey in nkeys]
            # self.R = cp.Parameter((nkey*3, nkey*3))
            # self.T = cp.Parameter((nkey*3,1))
            self.T = [np.tile(Ts[j].reshape(1,-1),nkey).reshape(-1,1) for j,nkey in enumerate(nkeys)]
            # print(self.T.value)
            
            R_es = [np.zeros((3*nkey, 3*nkey)) for nkey in nkeys]
            # Place the 3x3 matrix on the diagonal
            for j, nkey in enumerate(nkeys):
                for i in range(nkey):
                    R_es[j][i*3:(i+1)*3, i*3:(i+1)*3] = Rs[j]
            self.R = [R_e for R_e in R_es]
            # Constraints
            self.constraints = []
            
            for j, nkey in enumerate(nkeys):
                for i in range(0,nkey,3):
                    self.constraints += [self.z[j][i] == self.z[j][i+1], self.z[j][i] == self.z[j][i+2]]
            
                self.constraints += [self.xs[j] + self.Js[j]@self.dq == self.R[j]@cp.multiply(self.A[j],self.z[j]) + self.T[j] + self.deltas[j]]
            
                self.constraints += [z <= 10 for z in self.z[j]]
                self.constraints += [z >= 1 for z in self.z[j]]
            
            self.constraints += [-self.dq >= -1*(self.q_u.reshape(49,1)-self.q), self.dq >= -1*(self.q-self.q_l.reshape(49,1))]
            self.constraints += [self.dq_prev + self.dq >= self.dq_l, self.dq_prev + self.dq <= self.dq_u]
            
            # Problem
            to_minimize = cp.quad_form(self.dq,np.eye(self.dq.shape[0]))
            for j, nkey in enumerate(nkeys):
                to_minimize += cp.quad_form(self.deltas[j],np.eye(self.deltas[j].shape[0]))
                # # to_minimize += cp.quad_form(self.z[j],np.eye(self.z[j].shape[0]))
            
            self.obj = cp.Minimize(to_minimize)
            self.prob = cp.Problem(self.obj, self.constraints)
            self.qpIK_problems[key] = {"problem": self.prob, 
                                        #    "x_targets":self.x_targets,
                                        "xs" : self.xs,
                                        "Js" : self.Js,
                                        "deltas" : self.deltas,
                                        "dq_l" : self.dq_l,
                                        "dq_u" : self.dq_u,
                                        "dq_prev" : self.dq_prev,
                                        "dq" : self.dq,
                                        "q" : self.q,
                                        "z" : self.z,
                                        "A" : self.A,
                                        "R" : self.R,
                                        "T" : self.T,
                                            }
        else:
            # print(self.qpIK_problems[key].keys())
            self.prob = self.qpIK_problems[key]["problem"]
            self.xs = self.qpIK_problems[key]["xs"]
            self.Js = self.qpIK_problems[key]["Js"]
            self.deltas = self.qpIK_problems[key]["deltas"]
            self.dq_l = self.qpIK_problems[key]["dq_l"]
            self.dq_u = self.qpIK_problems[key]["dq_u"]
            self.dq_prev = self.qpIK_problems[key]["dq_prev"]
            self.dq = self.qpIK_problems[key]["dq"]
            self.q = self.qpIK_problems[key]["q"]
            self.z = self.qpIK_problems[key]["z"]
            self.A = self.qpIK_problems[key]["A"]
            self.R = self.qpIK_problems[key]["R"]
            self.T = self.qpIK_problems[key]["T"]
                
        self.dq_l.value = dt*self.qdot_l.reshape(49,1)
        self.dq_u.value = dt*self.qdot_u.reshape(49,1)
        self.dq_prev.value = np.zeros(self.q.shape)

        U = [np.hstack([kps2ds[j],np.ones((nkey,1))]).reshape(-1,1) for j,nkey in enumerate(nkeys)]
        C = [np.tile(np.array([Ks[j][0,2],Ks[j][1,2],0]),nkey).reshape(-1,1) for j,nkey in enumerate(nkeys)]
        F = [np.tile(np.array([Ks[j][0,0],Ks[j][1,1],1]),nkey).reshape(-1,1) for j,nkey in enumerate(nkeys)]

        for j,_ in enumerate(nkeys):
            self.A[j].value = (U[j]-C[j])/F[j]
                
        # older_loss = np.inf
        i = 0
        while i < max_iterations:
            self.q.value = self._nimble.getPositions().reshape(49,1)
            xs = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints))).reshape(3*len(self.joints),1)
            # print(self.xs.value.reshape(-1,3))
            J = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
            
            for j in range(len(nkeys)):
                mask3d = np.tile(masks[j][:,1].reshape(-1,1), (3, 1)).squeeze()
                # print(mask[j],mask3d)
                self.Js[j].value = J[mask3d,:]
                self.xs[j].value = xs[mask3d]
            
            self.prob.solve(warm_start=True,solver=cp.OSQP) #, warm_start=True,solver=cp.ECOS  verbose=True
            # print(i,self.prob.status,type(self.dq.value))
            # sol = self.R.value@(self.A.value*self.z.value) + self.T.value
            # print(sol.reshape(-1,3))
            # print("z:",self.z.value)
                        
            # if self.prob.status != "optimal":
            #     return
            
            # print(self.z.value)
            self.dq_prev.value += np.array(self.dq.value)
            self._nimble.setPositions(self.q.value+self.dq.value) # *0.01
            i+=1
      
    def filter(self,data_list=None,dt=100,Q=0.5,to_predict=True):
        if self.kf is None:
            self.qpIK(10,dt,precision=0.01) if data_list is None else self.multisource_qpIK(data_list,10,dt,precision=0.01)
            pos = self._nimble.getPositions()
            self.kf = [Kalman(dt,pos[i],Q) for i in range(pos.shape[0])]
        else:
            if to_predict:
                [kf.predict() for kf in self.kf]
            self.qpIK(10,dt,precision=0.01) if data_list is None else self.multisource_qpIK(data_list,10,dt,precision=0.01)
            
            pos = self._nimble.getPositions() # q from measurements
            
            for i in range(len(self.kf)):
                pos[i] = self.kf[i].update(pos[i],minval=self.q_l[i],maxval=self.q_u[i])
            self._nimble.setPositions(pos)
            
            
    def to_numpy(self):
        # return np.array(self._nimble.getJointWorldPositions(self.joints))
        return self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
        