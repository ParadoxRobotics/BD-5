
from dynamixel_sdk import *

# Position based servo-controller 
class ServoControllerBD5():
    def __init__(self, portHandler=None, packetHandler=None):
        # Constant
        self.PI = 3.14159265
        self.deg2rad = self.PI/180
        self.rad2deg = 180/self.PI
        self.rpm2rads = 0.10472

        # COM utils port + handler (2.0 in our case)
        self.portHandler = portHandler
        self.packetHandler = packetHandler

        # list of all joint IDs
        self.joints_ID = {
            "left_hip_yaw": 4, # XM430-W350-T
            "left_hip_roll": 6, # XM430-W350-T
            "left_hip_pitch": 8, # XM430-W350-T
            "left_knee": 10, # XM430-W350-T
            "left_ankle": 12, # XM430-W350-T
            "right_hip_yaw": 3, # XM430-W350-T
            "right_hip_roll": 5, # XM430-W350-T
            "right_hip_pitch": 7, # XM430-W350-T
            "right_knee": 9, # XM430-W350-T
            "right_ankle": 11, # XM430-W350-T
            "neck_pitch": 1, # XC430-W150-T
            "head_pitch": 2, # XC430-W150-T
        }
        self.joint_ID_list = list(self.joints_ID.values())
        self.joint_leg_ID_list = self.joint_ID_list[:10]

        # Joint limit in dxl value [min, max]
        self.joints_limit = {
            "left_hip_yaw": [1800,2295],
            "left_hip_roll": [1795,3275],
            "left_hip_pitch": [907,3188],
            "left_knee": [496,2048],
            "left_ankle": [816,3279],
            "right_hip_yaw": [1800,2295],
            "right_hip_roll": [822,2300],
            "right_hip_pitch": [907,3188],
            "right_knee": [496,2048],
            "right_ankle": [816,3279],
            "neck_pitch": [1195,2680],
            "head_pitch": [1145,2925],
        }
        self.joints_limit_list = list(self.joints_limit.values())

        # Angular correction for each joint
        self.joints_correction = {
            "left_hip_yaw": 1,
            "left_hip_roll": -1,
            "left_hip_pitch": -1,
            "left_knee": -1,
            "left_ankle": 1,
            "right_hip_yaw": 1,
            "right_hip_roll": 1,
            "right_hip_pitch": -1,
            "right_knee": -1,
            "right_ankle": 1,
            "neck_pitch": -1,
            "head_pitch": 1,
        }
        self.joints_correction_list = list(self.joints_correction.values())
        self.joints_leg_correction_list = self.joints_correction_list[:10]

        # Value limit XM430-W350-T and XC430-W150-T
        self.MAX_POS = 4096
        self.MAX_VEL = 1024 
        self.MAX_ANG = 2*self.PI

        # Mem address for XM430 and XC430
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_RETURN_DELAY_TIME = 9
        self.ADDR_POSITION_P_GAIN = 84
        self.ADDR_POSITION_I_GAIN = 82
        self.ADDR_POSITION_D_GAIN = 80
        self.ADDR_PROFILE_VELOCITY = 112
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_VOLTAGE = 144
        self.ADDR_PRESENT_POSITION = 132
        self.ADDR_PRESENT_VELOCITY = 128
        
        # Len return mem value for XM430 and XC430
        self.LEN_TORQUE_ENABLE = 1
        self.LEN_RETURN_DELAY_TIME = 1
        self.LEN_POSITION_P_GAIN = 2
        self.LEN_POSITION_I_GAIN = 2
        self.LEN_POSITION_D_GAIN = 2
        self.LEN_PROFILE_VELOCITY = 4
        self.LEN_GOAL_POSITION = 4
        self.LEN_PRESENT_VOLTAGE = 2
        self.LEN_PRESENT_POSITION = 4
        self.LEN_PRESENT_VELOCITY = 4
        
        # get sync group for read and write position 
        self.groupSyncWrite_pos = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION)
        self.groupSyncRead_pos = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)

        for ids in self.joint_leg_ID_list:
            # Add each motor to the bulk read list. 
            # Only the ID is needed here.
            addparam_result = self.groupSyncRead_pos.addParam(ids) # <-- CORRECTED LINE
            if not addparam_result:
                # Note: The original error message here was slightly misleading. Corrected it.
                raise Exception(f"[ID:{ids}] groupSyncRead_pos addparam failed")

    # correct rotation 
    def correctRotation(self, value, joint_list):
        corrected_value = []
        for i in range(len(value)):
            if value[i] != 0.0:
                corrected_value.append(value[i] * joint_list[i])
            else:
                corrected_value.append(value[i])
        return corrected_value

    # Clamp min-max joint value (in dxl angular space)
    def dxlClamp(self, value):
        return [max(l[0], min(v, l[1])) for v, l in zip(value, self.joints_limit_list)]
    
    # Position measure value in rad
    def dxl2position(self, value):
        angle_value = [(val - (self.MAX_POS/2)) * (self.MAX_ANG / (self.MAX_POS-1)) for val in value]
        return angle_value

    # Compute Goal position from rad value
    def position2dxl(self, value):
        dxl_values = [int((self.MAX_POS/2) + (val / self.MAX_ANG) * (self.MAX_POS-1)) for val in value]
        dxl_values = [max(0, min(self.MAX_POS - 1, val)) for val in dxl_values]
        return dxl_values

    # Check error
    def checkError(self, dxl_comm_result, dxl_error):
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            return False
        return True
    
    # Write to a single mem addr a value 
    def itemWrite(self, id, address, data, length):
        if length == 1:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, id, address, data)
        elif length == 2:
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, id, address, data)
        elif length == 4:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, id, address, data)
        else:
            print("Invalid data length...")
            return False
        return self.checkError(dxl_comm_result, dxl_error)

    # Write to mutliple mem addr a value 
    def itemWriteMultiple(self, ids, address, data, length):
        if not isinstance(data, list):
            for id in ids:
                success = self.itemWrite(id, address, data, length)
                if success != True:
                    return False
        else:
            for id, dat in zip(ids, data):
                success = self.itemWrite(id, address, dat, length)
                if success != True:
                    return False
        return True

    # Read eeprom register for a single servo
    def itemRead(self, id, address, length):
        if length == 1:
            state, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler, id, address)
        elif length == 2:
            state, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, id, address)
            if state > 0x7fff:
                state = state - 65536
        elif length == 4:
            state, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, id, address)
            if state > 0x7fffffff:
                state = state - 4294967296
        else:
            print("Invalid data length...")
            return 0, False
        return state, self.checkError(dxl_comm_result, dxl_error)

    # read eeprom register for multiple servos
    def itemReadMultiple(self, ids, address, length):
        states = []
        for id in ids:
            state, success = self.itemRead(id, address, length)
            if success != True:
                return [], False
            states.append(state)
        return states, True

    # Sync write to servos
    def syncWrite(self, groupSyncWrite, ids, commands, length):
        groupSyncWrite.clearParam()
        for id, cmd in zip(ids, commands):
            param = []
            if length == 4:
                param.append(DXL_LOBYTE(DXL_LOWORD(cmd)))
                param.append(DXL_HIBYTE(DXL_LOWORD(cmd)))
                param.append(DXL_LOBYTE(DXL_HIWORD(cmd)))
                param.append(DXL_HIBYTE(DXL_HIWORD(cmd)))
            elif length == 2:
                param.append(DXL_LOBYTE(cmd))
                param.append(DXL_HIBYTE(cmd))
            else:
                param.append(cmd)
            dxl_addparam_result = groupSyncWrite.addParam(id, param)
            if dxl_addparam_result != True:
                print("ID:%03d groupSyncWrite addparam failed" % id)
                return False
        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            return False
        return True

    # Sync read servos value
    def syncRead(self, groupSyncRead, ids, address, length):
        dxl_comm_result = groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            return [], False
        states = []
        for id in ids:                    
            if groupSyncRead.isAvailable(id, address, length):
                state = groupSyncRead.getData(id, address, length)
            else:
                print(f"[Warning] ID {id} has no available data (no status packet).")
                state = None
            if length == 2 and state > 0x7fff:
                state = state - 65536
            elif length == 4 and state > 0x7fffffff:
                state = state - 4294967296
            states.append(state)
        return states, True
    
    def ping(self):
        for id in self.joint_ID_list:
            model_num, dxl_comm_result, dxl_error = self.packetHandler.ping(self.portHandler, id)
            success = self.checkError(dxl_comm_result, dxl_error)
            if success:
                print("Pinged ID: %03d successfully! Model Number: %d" % (id, model_num))
            else:
                return False
        return True 
    
    # Enable torque 
    def enable_torque(self):
        self.itemWriteMultiple(self.joint_ID_list, self.ADDR_TORQUE_ENABLE, 1, self.LEN_TORQUE_ENABLE)

    # Disable torque 
    def disable_torque(self):
        self.itemWriteMultiple(self.joint_ID_list, self.ADDR_TORQUE_ENABLE, 0, self.LEN_TORQUE_ENABLE)

    # Set return delay for all servos
    def set_return_delay(self, value):
        self.itemWriteMultiple(self.joint_ID_list, self.ADDR_RETURN_DELAY_TIME, value, self.LEN_RETURN_DELAY_TIME)
        print(f"Setting Return Delay Time for all servos to {value*2} us...")

    # Set P gain
    def set_P_gain(self, ids, value):
        self.itemWriteMultiple(ids, self.ADDR_POSITION_P_GAIN, value, self.LEN_POSITION_P_GAIN)
        
    # Set I gain
    def set_I_gain(self, ids, value):
        self.itemWriteMultiple(ids, self.ADDR_POSITION_I_GAIN, value, self.LEN_POSITION_I_GAIN)
        
    # Set D gain
    def set_D_gain(self, ids, value):
        self.itemWriteMultiple(ids, self.ADDR_POSITION_D_GAIN, value, self.LEN_POSITION_D_GAIN)

    # Set PID for the legs
    def set_PID(self, pid):
        self.set_P_gain(ids=self.joint_leg_ID_list, value=[pid[0]] * len(self.joint_leg_ID_list))
        self.set_I_gain(ids=self.joint_leg_ID_list, value=[pid[1]] * len(self.joint_leg_ID_list))
        self.set_D_gain(ids=self.joint_leg_ID_list, value=[pid[2]] * len(self.joint_leg_ID_list))

    # Set goal position to all servos 
    def set_position(self, value):
        # correct rotation
        value = self.correctRotation(value, self.joints_correction_list)
        # convert value to dxl 
        ang_pos = self.position2dxl(value=value)
        # Clamp value 
        ang_pos = self.dxlClamp(value=ang_pos)
        # send command
        self.syncWrite(self.groupSyncWrite_pos, self.joint_ID_list, ang_pos, self.LEN_GOAL_POSITION)
        
    # Get current position (only the legs)
    def get_position(self):
        # read raw position 
        positions, success = self.syncRead(self.groupSyncRead_pos, self.joint_leg_ID_list, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
        # convert to radian 
        positions = self.dxl2position(value=positions)
        # correct rotation
        positions = self.correctRotation(positions, self.joints_leg_correction_list)
        return positions, success

if __name__=='__main__':   
    import time 
    import numpy as np
    from Gamepad import Gamepad

    # Time param
    command_freq = 20
    ctrl_freq = 50
    ctrl_dt = 1.0 / ctrl_freq

    # Init gamepad
    controller = Gamepad(command_freq=command_freq, vel_range_x=[-0.6, 0.6], vel_range_y=[-0.6, 0.6], vel_range_rot=[-1.0, 1.0], head_range=[-0.5236, 0.5236], deadzone=0.02)

    # Dxl param
    port = "/dev/ttyUSB0"
    baudrate = 1000000
    # connect to U2D2
    portHandler = PortHandler(port)
    packetHandler = PacketHandler(2.0)
    # check connection 
    if portHandler.openPort():
        print("Successfully opened the port at %s!" % port)
    else:
        portHandler.closePort()
        raise Exception("Failed to open the port at %s!", port)
    if portHandler.setBaudRate(baudrate):
        print("Succeeded to change the baudrate to %d bps!" % baudrate)
    else:
        portHandler.closePort()
        raise Exception("Failed to change the baudrate to %d bps!" % baudrate)
    # init servos class
    BDX = ServoControllerBD5(portHandler=portHandler, packetHandler=packetHandler)
    # ping all servos 
    if BDX.ping():
        print("Servos ready !")
    else:
        portHandler.closePort()
        raise Exception("Error in servos ID or state !")
    
    # set default angles
    default_angles_leg = [0.0,
                          -0.0523599,
                          0.8725738534323367,
                          1.7451477068646735,
                          0.8725738534323367,
                          0.0,
                          -0.0523599,
                          0.8725738534323367,
                          1.7451477068646735,
                          0.8725738534323367]
    default_angles_head = [0.5306, -0.5306]
    default_angles_full = default_angles_leg + default_angles_head
    zeros_position = [0.0] * len(default_angles_full)

    while True:
        last_state, head_t, S_pressed, T_pressed, C_pressed, X_pressed = controller.get_last_command()
        if C_pressed == True:
            print("BD-5 ACTIVATE !")
            break

    # Activate + Set default angles
    BDX.set_return_delay(value=10)
    BDX.set_PID(pid=[800, 0, 0])
    BDX.enable_torque()
    BDX.set_position(default_angles_full)
    
    i=0
    try:
        print("Starting")
        start_t = time.time()
        while True:
            t = time.time()
            last_state, head_t, S_pressed, T_pressed, C_pressed, X_pressed = controller.get_last_command()
            if X_pressed == True:
                print("Kill switch pressed !")
                break
            pos, state = BDX.get_position()
            BDX.set_position(default_angles_full)
            #print(pos)
            # time control 
            i+=1
            took = time.time() - t
            if (1 / ctrl_freq - took) < 0:
                print(
                    "Policy control budget exceeded by",
                    np.around(took - 1 / ctrl_freq, 3),
                )
            time.sleep(max(0, 1 / ctrl_freq - took))
            
        time.sleep(1)

        BDX.disable_torque()
        portHandler.closePort()
        print("Port closed !")
        exit()

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected !")
        BDX.disable_torque()
        portHandler.closePort()
        print("Port closed !")
        raise


