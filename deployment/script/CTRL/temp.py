import os
import time
import math
from dynamixel_sdk import *

# --- Platform-specific setup for getch ---
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

class DynamixelController:
    """
    A class to control a set of Dynamixel servos for a biped robot.
    Handles communication, unit conversions, and bulk operations.
    """
    # --- DYNAMIXEL Constants ---
    PROTOCOL_VERSION = 2.0
    
    # Control Table Addresses for XM430-W350 and XC430-W150 (common addresses)
    ADDR_TORQUE_ENABLE      = 64
    ADDR_GOAL_POSITION      = 116
    ADDR_PRESENT_POSITION   = 132
    ADDR_PRESENT_VELOCITY   = 128
    ADDR_POSITION_D_GAIN    = 80
    ADDR_POSITION_I_GAIN    = 82
    ADDR_POSITION_P_GAIN    = 84

    # Data Length
    LEN_GOAL_POSITION       = 4
    LEN_PRESENT_POSITION    = 4
    LEN_PRESENT_VELOCITY    = 4
    LEN_PID_GAIN            = 2 # P, I, and D gains are 2 bytes each

    # Unit Conversions
    # XM430-W350 has a resolution of 4096 steps per revolution (2*pi radians)
    DXL_POS_TO_RAD = (2 * math.pi) / 4096
    # From e-manual, velocity unit is 0.229 RPM. 1 RPM = 2*pi/60 rad/s
    DXL_VEL_TO_RAD_S = 0.229 * (2 * math.pi) / 60
    
    def __init__(self, device_name: str, baudrate: int, joint_config: dict):
        """
        Initializes the DynamixelController.

        Args:
            device_name (str): The port name (e.g., '/dev/ttyUSB0' or 'COM3').
            baudrate (int): The communication baudrate.
            joint_config (dict): A dictionary containing 'joints_ID' and 'joints_correction'.
        """
        self.portHandler = PortHandler(device_name)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        # Store robot configuration
        self.joints_ID = joint_config["joints_ID"]
        self.joints_correction = joint_config["joints_correction"]
        self.joint_ids_list = list(self.joints_ID.values())
        self.id_to_name = {v: k for k, v in self.joints_ID.items()}

        # Initialize bulk/sync communication handlers
        self.groupSyncWrite_position = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION)
        self.groupBulkRead_state = GroupBulkRead(self.portHandler, self.packetHandler)
        self.groupBulkWrite_pid = GroupBulkWrite(self.portHandler, self.packetHandler)
        
        # Open port and set baudrate
        if not self.portHandler.openPort():
            raise IOError(f"Failed to open port {device_name}")
        if not self.portHandler.setBaudRate(baudrate):
            raise IOError(f"Failed to set baudrate to {baudrate}")
            
        print(f"Successfully connected to Dynamixels on {device_name} at {baudrate} baud.")
        
        # Pre-register all servos for bulk reading position and velocity
        for dxl_id in self.joint_ids_list:
            self.groupBulkRead_state.addParam(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
            self.groupBulkRead_state.addParam(dxl_id, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY)

    def _check_comm_result(self, dxl_comm_result: int, dxl_error: int):
        """A helper to check and print communication results."""
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Communication Error: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return False
        if dxl_error != 0:
            print(f"Packet Error: {self.packetHandler.getRxPacketError(dxl_error)}")
            return False
        return True

    def ping_servos(self):
        """Pings all servos and prints the IDs of those that respond."""
        print("Pinging all servos...")
        dxl_comm_result, dxl_error = self.packetHandler.broadcastPing(self.portHandler)
        if not self._check_comm_result(dxl_comm_result, dxl_error):
            print("Broadcast Ping failed.")
            return

        found_ids = [dxl_id for dxl_id in self.joint_ids_list if self.packetHandler.getBroadcastPingResult(dxl_id)]
        
        print(f"Found {len(found_ids)} servos.")
        if found_ids:
            print("Responding Servo IDs:", sorted(found_ids))
        
        missing_ids = set(self.joint_ids_list) - set(found_ids)
        if missing_ids:
            print("WARNING: Did not find expected servos with IDs:", sorted(list(missing_ids)))

    def _set_torque_for_all(self, enable: bool):
        """Internal helper to enable or disable torque for all joints."""
        groupBulkWrite_torque = GroupBulkWrite(self.portHandler, self.packetHandler)
        val = 1 if enable else 0
        action = "Enabling" if enable else "Disabling"
        print(f"{action} torque for all servos...")

        for dxl_id in self.joint_ids_list:
            param = [DXL_LOBYTE(val), DXL_HIBYTE(val)] # Torque enable is 1 byte, but bulk write is flexible
            groupBulkWrite_torque.addParam(dxl_id, self.ADDR_TORQUE_ENABLE, 1, [val])

        dxl_comm_result = groupBulkWrite_torque.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Torque {action} failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print("Torque set successfully.")

    def _set_torque_for_all(self, enable: bool):
        """Internal helper to enable or disable torque for all joints using GroupBulkWrite."""
        
        groupBulkWrite_torque = GroupBulkWrite(self.portHandler, self.packetHandler)
        
        # The value to write: 1 for enable, 0 for disable.
        torque_value = 1 if enable else 0
        action_str = "Enabling" if enable else "Disabling"
        print(f"{action_str} torque for all servos...")

        # Torque Enable is a 1-byte register. The data must be passed as a list/array of bytes.
        data_to_write = [torque_value]
        data_length = 1 

        # Add each servo to the bulk write instruction
        for dxl_id in self.joint_ids_list:
            # addParam returns True on success, False on failure (e.g., buffer full)
            success = groupBulkWrite_torque.addParam(dxl_id, 
                                                     self.ADDR_TORQUE_ENABLE, 
                                                     data_length, 
                                                     data_to_write)
            if not success:
                print(f"Failed to add param for torque control on ID {dxl_id}")
                return # Abort if we can't add the parameter

        # Transmit the entire packet
        dxl_comm_result = groupBulkWrite_torque.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Torque {action_str} failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        else:
            print("Torque set successfully.")

    def enable_torque(self):
        """Enables torque for all configured servos."""
        self._set_torque_for_all(True)

    def disable_torque(self):
        """Disables torque for all configured servos."""
        self._set_torque_for_all(False)

    def read_state(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Reads the present position (rad) and velocity (rad/s) of all servos.

        Returns:
            A tuple containing two dictionaries:
            - (positions_rad, velocities_rad_s)
            - Returns empty dictionaries on failure.
        """
        positions_rad = {}
        velocities_rad_s = {}
        
        dxl_comm_result = self.groupBulkRead_state.txrxPacket()
        if not self._check_comm_result(dxl_comm_result, 0):
            return {}, {}

        for dxl_id in self.joint_ids_list:
            joint_name = self.id_to_name[dxl_id]
            correction = self.joints_correction[joint_name]

            # Check if data is available
            if not self.groupBulkRead_state.isAvailable(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION):
                print(f"Warning: No position data for ID {dxl_id} ({joint_name})")
                continue
            if not self.groupBulkRead_state.isAvailable(dxl_id, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY):
                print(f"Warning: No velocity data for ID {dxl_id} ({joint_name})")
                continue

            # Read raw DXL values
            dxl_pos = self.groupBulkRead_state.getData(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
            dxl_vel = self.groupBulkRead_state.getData(dxl_id, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY)
            
            # Convert to radians and rad/s, applying correction
            # Center position is 2048. We subtract this to make 0 the center.
            # Convert signed velocity value
            if dxl_vel > 2147483647: dxl_vel = dxl_vel - 4294967296
            
            positions_rad[joint_name] = (dxl_pos - 2048) * self.DXL_POS_TO_RAD * correction
            velocities_rad_s[joint_name] = dxl_vel * self.DXL_VEL_TO_RAD_S * correction
            
        return positions_rad, velocities_rad_s

    def set_goal_positions_rad(self, goal_positions_rad: dict[str, float]):
        """
        Sets the goal position for multiple servos using GroupSyncWrite.

        Args:
            goal_positions_rad (dict): A dictionary mapping joint names to goal positions in radians.
        """
        self.groupSyncWrite_position.clearParam()
        
        for joint_name, goal_rad in goal_positions_rad.items():
            if joint_name not in self.joints_ID:
                print(f"Warning: Joint '{joint_name}' not in configuration. Skipping.")
                continue

            dxl_id = self.joints_ID[joint_name]
            correction = self.joints_correction[joint_name]
            
            # Convert radians back to DXL position value
            dxl_goal_pos = int(2048 + (goal_rad / self.DXL_POS_TO_RAD) * correction)
            
            # Pack value into 4 bytes (little-endian)
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(dxl_goal_pos)), DXL_HIBYTE(DXL_LOWORD(dxl_goal_pos)),
                                   DXL_LOBYTE(DXL_HIWORD(dxl_goal_pos)), DXL_HIBYTE(DXL_HIWORD(dxl_goal_pos))]
            
            self.groupSyncWrite_position.addParam(dxl_id, param_goal_position)
            
        dxl_comm_result = self.groupSyncWrite_position.txPacket()
        if not self._check_comm_result(dxl_comm_result, 0):
            print("Failed to set goal positions.")

    def set_pid_gains(self, P: int, I: int, D: int, joint_names: list[str] = None):
        """
        Sets the Position P, I, and D gains for specified servos.

        Args:
            P (int): Proportional gain (0-16383).
            I (int): Integral gain (0-16383).
            D (int): Derivative gain (0-16383).
            joint_names (list[str], optional): List of joints to apply gains to.
                                                If None, applies to all joints. Defaults to None.
        """
        self.groupBulkWrite_pid.clearParam()
        target_ids = [self.joints_ID[name] for name in joint_names] if joint_names else self.joint_ids_list
        
        print(f"Setting PID gains (P={P}, I={I}, D={D}) for {len(target_ids)} servos...")
        
        param_p = [DXL_LOBYTE(P), DXL_HIBYTE(P)]
        param_i = [DXL_LOBYTE(I), DXL_HIBYTE(I)]
        param_d = [DXL_LOBYTE(D), DXL_HIBYTE(D)]

        for dxl_id in target_ids:
            self.groupBulkWrite_pid.addParam(dxl_id, self.ADDR_POSITION_P_GAIN, self.LEN_PID_GAIN, param_p)
            self.groupBulkWrite_pid.addParam(dxl_id, self.ADDR_POSITION_I_GAIN, self.LEN_PID_GAIN, param_i)
            self.groupBulkWrite_pid.addParam(dxl_id, self.ADDR_POSITION_D_GAIN, self.LEN_PID_GAIN, param_d)

        dxl_comm_result = self.groupBulkWrite_pid.txPacket()
        if not self._check_comm_result(dxl_comm_result, 0):
            print("Failed to set PID gains.")
        else:
            print("PID gains set successfully.")
            
    def close_port(self):
        """Disables torque and closes the serial port."""
        print("Closing port...")
        self.disable_torque()
        time.sleep(0.2) # Give a moment for the command to execute
        self.portHandler.closePort()
        print("Port closed.")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# new code 
# ----------------------------------------------------------------------------------------------------------------------------------------------

import os
import math
import time
from dynamixel_sdk import * # Imports all Dynamixel SDK constants and methods

# --- Platform-specific setup for getch (for the example)
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
# ---

class BipedController:
    """
    A high-performance controller for a biped robot using Dynamixel servos.
    This class handles low-level communication and provides a high-level interface
    for controlling the robot's joints in radians.
    """

    # --- Constants for Dynamixel XM/XC Series ---
    # Control Table Addresses
    ADDR_TORQUE_ENABLE      = 64
    ADDR_POSITION_D_GAIN    = 80
    ADDR_POSITION_I_GAIN    = 82
    ADDR_POSITION_P_GAIN    = 84
    ADDR_GOAL_POSITION      = 116
    ADDR_PRESENT_POSITION   = 132

    # Data Lengths
    LEN_GOAL_POSITION       = 4  # bytes
    LEN_PRESENT_POSITION    = 4  # bytes
    LEN_PID_GAIN            = 2  # bytes

    # Protocol and Baudrate
    PROTOCOL_VERSION        = 2.0
    BAUDRATE                = 1000000  # Example: 1 Mbps, adjust if needed

    # Radian to DXL Value Conversion
    # XM430/XC430 have 4096 steps per revolution (2*pi radians)
    # The zero position (0 rad) corresponds to DXL value 2048.
    DXL_PER_RADIAN = 4096 / (2 * math.pi)
    DXL_ZERO_OFFSET = 2048

    def __init__(self, device_name):
        """
        Initializes the BipedController.

        Args:
            device_name (str): The port name for the U2D2 or USB2AX (e.g., "/dev/ttyUSB0" on Linux, "COM3" on Windows).
        
        Raises:
            IOError: If the port cannot be opened or the baudrate cannot be set.
        """
        # --- Robot Configuration ---
        self.joints_ID = {
            "left_hip_yaw": 4, "left_hip_roll": 6, "left_hip_pitch": 8,
            "left_knee": 10, "left_ankle": 12, "right_hip_yaw": 3,
            "right_hip_roll": 5, "right_hip_pitch": 7, "right_knee": 9,
            "right_ankle": 11, "neck_pitch": 1, "head_pitch": 2
        }
        self.joint_names = list(self.joints_ID.keys())
        self.joint_ids = list(self.joints_ID.values())

        self.joints_correction = {
            "left_hip_yaw": 1, "left_hip_roll": -1, "left_hip_pitch": -1,
            "left_knee": -1, "left_ankle": 1, "right_hip_yaw": 1,
            "right_hip_roll": 1, "right_hip_pitch": -1, "right_knee": -1,
            "right_ankle": 1, "neck_pitch": -1, "head_pitch": 1
        }
        
        # --- SDK Initialization ---
        self.portHandler = PortHandler(device_name)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        # --- SDK Group Handlers for Performance ---
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION)
        self.groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)
        
        # Open port and set baudrate
        self._open_port()
        self._set_baudrate()

        print("BipedController initialized successfully.")

    def _open_port(self):
        if not self.portHandler.openPort():
            raise IOError(f"Failed to open the port: {self.portHandler.getPortName()}")
        print(f"Port '{self.portHandler.getPortName()}' opened.")

    def _set_baudrate(self):
        if not self.portHandler.setBaudRate(self.BAUDRATE):
            raise IOError(f"Failed to set the baudrate to {self.BAUDRATE}")
        print(f"Baudrate set to {self.BAUDRATE}.")

    def _check_comm_result(self, dxl_comm_result, dxl_error):
        """Checks communication result and raises an exception if there's an error."""
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(f"Communication failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            raise RuntimeError(f"Dynamixel error: {self.packetHandler.getRxPacketError(dxl_error)}")

    # --- Unit Conversion Helpers ---
    def _rad_to_dxl(self, joint_name, rad):
        """Converts radians to Dynamixel position value, applying joint correction."""
        correction = self.joints_correction.get(joint_name, 1)
        # The center position (0 rad) is 2048.
        # Positive radians turn one way, negative the other.
        # We apply the correction factor to match the desired direction.
        pos = self.DXL_ZERO_OFFSET + (correction * rad * self.DXL_PER_RADIAN)
        return int(max(0, min(4095, pos))) # Clamp between 0 and 4095

    def _dxl_to_rad(self, joint_name, dxl_val):
        """Converts Dynamixel position value to radians, applying joint correction."""
        correction = self.joints_correction.get(joint_name, 1)
        # Reverse the conversion formula
        rad = (dxl_val - self.DXL_ZERO_OFFSET) / self.DXL_PER_RADIAN
        return correction * rad
        
    # --- Public API Methods ---
    def ping_servos(self):
        """Pings all servos defined in the configuration and prints their status."""
        print("Pinging servos...")
        found_servos = []
        for joint_name, joint_id in self.joints_ID.items():
            dxl_model_number, dxl_comm_result, dxl_error = self.packetHandler.ping(self.portHandler, joint_id)
            if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                print(f"[ID:{joint_id:03d}] {joint_name:<15} ... SUCCESS (Model: {dxl_model_number})")
                found_servos.append(joint_id)
            else:
                print(f"[ID:{joint_id:03d}] {joint_name:<15} ... FAILED")
        return found_servos

    def set_torque(self, state):
        """
        Enables or disables torque for all servos.

        Args:
            state (bool): True to enable torque, False to disable.
        """
        mode = 1 if state else 0
        mode_str = "Enabling" if state else "Disabling"
        print(f"{mode_str} torque for all servos...")
        for joint_id in self.joint_ids:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, joint_id, self.ADDR_TORQUE_ENABLE, mode)
            self._check_comm_result(dxl_comm_result, dxl_error)
        print("Torque set successfully.")

    def set_poses_rad(self, poses_rad: dict):
        """
        Sets the goal position for multiple servos simultaneously using GroupSyncWrite.

        Args:
            poses_rad (dict): A dictionary mapping joint names (str) to positions (float, in radians).
                              e.g., {"left_hip_pitch": 0.5, "right_hip_pitch": -0.5}
        """
        self.groupSyncWrite.clearParam()
        for name, rad in poses_rad.items():
            if name not in self.joints_ID:
                print(f"Warning: Joint '{name}' not found in configuration. Skipping.")
                continue
            
            dxl_id = self.joints_ID[name]
            dxl_goal_position = self._rad_to_dxl(name, rad)
            
            # Allocate 4 bytes for the goal position value
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(dxl_goal_position)),
                                   DXL_HIBYTE(DXL_LOWORD(dxl_goal_position)),
                                   DXL_LOBYTE(DXL_HIWORD(dxl_goal_position)),
                                   DXL_HIBYTE(DXL_HIWORD(dxl_goal_position))]
            
            # Add parameter to the sync write storage
            addparam_result = self.groupSyncWrite.addParam(dxl_id, param_goal_position)
            if not addparam_result:
                raise RuntimeError(f"Failed to add param for DXL ID {dxl_id}")

        # Transmit the packet
        dxl_comm_result = self.groupSyncWrite.txPacket()
        self._check_comm_result(dxl_comm_result, 0) # Error is not reported by txPacket

    def read_poses_rad(self):
        """
        Reads the present position of all servos simultaneously using GroupBulkRead.

        Returns:
            dict: A dictionary mapping joint names (str) to their current positions (float, in radians).
                  Returns an empty dict on failure.
        """
        self.groupBulkRead.clearParam()
        
        # Add all servos to the bulk read list
        for joint_id in self.joint_ids:
            addparam_result = self.groupBulkRead.addParam(joint_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
            if not addparam_result:
                print(f"Failed to add param for bulk read on DXL ID {joint_id}")
                return {}

        # Transmit the bulk read instruction packet
        dxl_comm_result = self.groupBulkRead.txRxPacket()
        self._check_comm_result(dxl_comm_result, 0) # Error is not reported by txRxPacket

        # Retrieve and process the data
        poses = {}
        for name, joint_id in self.joints_ID.items():
            # Check if data for this servo is available
            is_available = self.groupBulkRead.isAvailable(joint_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
            if not is_available:
                print(f"Warning: No data received for DXL ID {joint_id} ({name})")
                continue

            # Get the raw DXL position value
            dxl_present_position = self.groupBulkRead.getData(joint_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
            
            # Convert to radians and store
            poses[name] = self._dxl_to_rad(name, dxl_present_position)
            
        return poses

    def set_pid_gains(self, joint_name: str, p: int, i: int, d: int):
        """
        Sets the PID gains for a specific servo.
        NOTE: Torque should typically be disabled before changing PID gains.

        Args:
            joint_name (str): The name of the joint to configure.
            p (int): The Proportional gain (0-16383).
            i (int): The Integral gain (0-16383).
            d (int): The Derivative gain (0-16383).
        """
        if joint_name not in self.joints_ID:
            raise ValueError(f"Joint '{joint_name}' not found in configuration.")
        
        dxl_id = self.joints_ID[joint_name]
        
        print(f"Setting PID for {joint_name} (ID: {dxl_id}) to P={p}, I={i}, D={d}")

        # Temporarily disable torque for this servo
        self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0)
        
        # Write PID values (2 bytes each)
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, self.ADDR_POSITION_P_GAIN, p)
        self._check_comm_result(dxl_comm_result, dxl_error)
        
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, self.ADDR_POSITION_I_GAIN, i)
        self._check_comm_result(dxl_comm_result, dxl_error)
        
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, self.ADDR_POSITION_D_GAIN, d)
        self._check_comm_result(dxl_comm_result, dxl_error)
        
        # Re-enable torque
        self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 1)

        print("PID gains set successfully.")

    def close(self):
        """Disables torque on all servos and closes the serial port."""
        print("\nClosing BipedController...")
        try:
            self.set_torque(False)
        except RuntimeError as e:
            print(f"Could not disable torque during close: {e}")
        finally:
            self.portHandler.closePort()
            print("Port closed.")
            
    # --- Context Manager for robust resource handling ---
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# --- Example Usage ---
if __name__ == '__main__':
    # !!! IMPORTANT: Set your device name here !!!
    # Linux:   "/dev/ttyUSB0"
    # Windows: "COM3"
    # Mac:     "/dev/tty.usbserial-*"
    DEVICE_NAME = "/dev/ttyUSB0" 
    
    try:
        # The 'with' statement ensures robot.close() is called automatically
        with BipedController(DEVICE_NAME) as robot:
            
            print("\n--- Pinging Servos ---")
            robot.ping_servos()
            
            print("\n--- Enabling Torque ---")
            robot.set_torque(True)

            print("\n--- Reading Initial Pose ---")
            initial_poses = robot.read_poses_rad()
            for name, pos in initial_poses.items():
                print(f"{name:<15}: {pos:.3f} rad")
            
            print("\n--- Setting PID for one joint (example) ---")
            # These are example values, you will need to tune them
            # robot.set_pid_gains("left_knee", p=840, i=0, d=0)

            print("\n--- Moving to Home Position (0 radians) ---")
            home_pose = {name: 0.0 for name in robot.joint_names}
            robot.set_poses_rad(home_pose)
            time.sleep(2) # Give time for the robot to move

            print("\n--- Performing a simple sine wave motion on hips ---")
            print("Press any key to stop.")
            start_time = time.time()
            while True:
                # Check for key press to exit loop
                if os.name == 'nt' and msvcrt.kbhit():
                    break
                elif os.name != 'nt' and select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    break

                elapsed_time = time.time() - start_time
                # Move hips in a sine wave, amplitude ~20 degrees (0.35 rad)
                angle = 0.35 * math.sin(elapsed_time * 2) # frequency of 2 rad/s

                # Create the pose dictionary for the command
                target_pose = {
                    "left_hip_pitch": angle,
                    "right_hip_pitch": angle
                }

                # Set the pose
                robot.set_poses_rad(target_pose)

                # Read and print current pose (optional, slows down the loop)
                current_poses = robot.read_poses_rad()
                lp_pos = current_poses.get("left_hip_pitch", 0)
                rp_pos = current_poses.get("right_hip_pitch", 0)
                print(f"\rTarget: {angle:.3f} | Current L:{lp_pos:.3f} R:{rp_pos:.3f}", end="")

                time.sleep(0.02) # Control loop frequency ~50Hz

    except (IOError, RuntimeError) as e:
        print(f"\nAn error occurred: {e}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    
    print("\nExample finished.")
