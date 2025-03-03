import numpy as np 
import math 

class BDXWalkingController:
    def __init__(self, dHipYawRoll, dHipRollPitch, dLegSection, dFootSole):
        # dimension variable 
        self.LH1 = dHipYawRoll # distance between hip yaw and roll axis
        self.LH2 = dHipRollPitch # distance between hip roll and pitch axis 
        self.LT = dLegSection # thigh size -> identical to calve
        self.LC = dLegSection # calve size -> identical to thigh
        self.LF = dFootSole # distance between the joint and sole 
        # joint variable 
        self.joints = {
            "right_hip_yaw": 0,
            "right_hip_roll": 0,
            "right_hip_pitch": 0,
            "right_knee": 0,
            "right_ankle": 0,
            "left_hip_yaw": 0,
            "left_hip_roll": 0,
            "left_hip_pitch": 0,
            "left_knee": 0,
            "left_ankle": 0,
            "neck_pitch": 0,
            "head_pitch": 0,
        }
        
    # Leg inverse kinemtics
    def IK_leg(self, x, y, z, theta):
        # yaw axis compensation for turning
        yaw = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2) # turn radius
        y = r*np.sin(yaw+theta)
        x = r*np.cos(yaw+theta)
        # Compute local distance between hip pitch and ankle joints
        LL = np.sqrt(x**2 + (np.sqrt(y**2 + z**2) - self.LF - self.LH2)**2)
        # Compute hip and knee pitch 
        hip_pitch = np.arccos(LL / (2*self.LT))
        knee_pitch = np.arcsin(LL / (2*self.LC))
        # Compute pitch axis -> constraint: sole // floor with respect to the body 
        sigma = np.arcsin(x/LL)
        # Compute angle for hip, knee and ankle on the sigital plane 
        theta_hip_pitch = (hip_pitch + sigma) * 57.3
        theta_knee = (2*knee_pitch) * 57.3
        theta_ankle = (hip_pitch - sigma) * 57.3
        # Compute hip roll
        theta_hip_roll = np.arctan2(y, z) * 57.3
        # Compute hip yaw 
        theta_hip_yaw = (-theta) * 57.3
        # return angle (degree)
        return theta_hip_yaw, theta_hip_roll, theta_hip_pitch, theta_knee, theta_ankle
    
    # Walking Controller 
    def Walking_controller(self, Amplitude, NbStep, Sample, LegStride, LegHeight, InitLegLength):
        for j in range(0, NbStep):
            for i in range(0, Sample):
                # Lateral trajectory
                hip_y = Amplitude * np.sin(np.pi * 2 * i / Sample) # lateral phase
                # Stride trajectory
                right_x = LegStride * np.sin(np.pi * 2 * i / Sample - np.pi / 2) # phase (-90°)
                left_x = LegStride * np.sin(np.pi * 2 * i / Sample + np.pi / 2) # phase (+90°)
                # Foot lifting trajectory
                right_z = InitLegLength - LegHeight * np.sin(np.pi * 2 * i / Sample) # phase (-180°)
                left_z = InitLegLength + LegHeight * np.sin(np.pi * 2 * i / Sample) # phase (+180°)
                # Apply z position only for foot lifting
                if (right_z > InitLegLength):
                    right_z = InitLegLength
                if (left_z > InitLegLength):
                    left_z = InitLegLength
                # Compute IK 
                rhy, rhr, rhp, rk, ra = self.IK_leg(X=right_x, y=hip_y, z=right_z, theta=0)
                hy, lhr, lhp, lk, la = self.IK_leg(X=left_x, y=hip_y, z=left_z, theta=0)
        return None