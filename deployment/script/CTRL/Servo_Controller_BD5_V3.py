      
import serial
import time
import numpy as np
from Gamepad import Gamepad # Assuming Gamepad.py is in the same directory

# =============================================================================
#  1. LOW-LEVEL I/O CLASS (Optimized for Bulk Communication)
# =============================================================================
class DynamixelBulkControl:
    """
    An optimized class for high-frequency control of MULTIPLE Dynamixel servos
    using Protocol 2.0. Replaces the need for the official SDK for core tasks.
    """
    # Protocol and Instruction Constants
    HEADER = b'\xFF\xFF\xFD\x00'
    INST_PING = 0x01
    INST_WRITE = 0x03
    INST_SYNC_READ = 0x82
    INST_BULK_WRITE = 0x93
    BROADCAST_ID = 0xFE

    # Pre-computed CRC-16 table for performance
    CRC_TABLE = [
    0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011,
    0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027, 0x0022,
    0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072,
    0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041,
    0x80C3, 0x00C6, 0x00CC, 0x80C9, 0x00D8, 0x80DD, 0x80D7, 0x00D2,
    0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1,
    0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1,
    0x8093, 0x0096, 0x009C, 0x8099, 0x0088, 0x808D, 0x8087, 0x0082,
    0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192,
    0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1,
    0x01E0, 0x81E5, 0x81EF, 0x01EA, 0x81FB, 0x01FE, 0x01F4, 0x81F1,
    0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2,
    0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151,
    0x8173, 0x0176, 0x017C, 0x8179, 0x0168, 0x816D, 0x8167, 0x0162,
    0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
    0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101,
    0x8303, 0x0306, 0x030C, 0x8309, 0x0318, 0x831D, 0x8317, 0x0312,
    0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321,
    0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371,
    0x8353, 0x0356, 0x035C, 0x8359, 0x0348, 0x834D, 0x8347, 0x0342,
    0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1,
    0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2,
    0x83A3, 0x03A6, 0x03AC, 0x83A9, 0x03B8, 0x83BD, 0x83B7, 0x03B2,
    0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381,
    0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291,
    0x82B3, 0x02B6, 0x02BC, 0x82B9, 0x02A8, 0x82AD, 0x82A7, 0x02A2,
    0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2,
    0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1,
    0x8243, 0x0246, 0x024C, 0x8249, 0x0258, 0x825D, 0x8257, 0x0252,
    0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261,
    0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231,
    0x8213, 0x0216, 0x021C, 0x8219, 0x0208, 0x820D, 0x8207, 0x0202
    ]

    def __init__(self, port, motor_ids, baudrate=1000000):
        self.port = port
        self.motor_ids = motor_ids
        self.baudrate = baudrate
        self.serial = None

    def open_port(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.02)
            return True
        except serial.SerialException as e:
            print(f"Failed to open port {self.port}: {e}")
            return False

    def close_port(self):
        if self.serial and self.serial.is_open:
            self.serial.close()

    def _calculate_crc(self, data):
        crc = 0
        for byte in data:
            crc = (crc << 8) ^ self.CRC_TABLE[((crc >> 8) ^ byte) & 0xFF]
        return crc & 0xFFFF

    def _create_packet(self, motor_id, instruction, parameters=b''):
        length = len(parameters) + 3
        packet = bytearray(self.HEADER)
        packet.extend([motor_id, length & 0xFF, (length >> 8) & 0xFF, instruction])
        packet.extend(parameters)
        crc = self._calculate_crc(packet)
        packet.extend([crc & 0xFF, (crc >> 8) & 0xFF])
        return bytes(packet)

    def ping(self, motor_id):
        packet = self._create_packet(motor_id, self.INST_PING)
        self.serial.write(packet)
        response = self.serial.read(14) # Read response for one motor
        if len(response) >= 14 and response[0:4] == self.HEADER and response[8] == 0:
            model_num = int.from_bytes(response[11:13], 'little')
            return True, model_num
        return False, 0

    def sync_read(self, address, length):
        params = address.to_bytes(2, 'little') + length.to_bytes(2, 'little')
        for motor_id in self.motor_ids:
            params += bytes([motor_id])
        
        packet = self._create_packet(self.BROADCAST_ID, self.INST_SYNC_READ, params)
        self.serial.write(packet)
        
        # Expected length: 11 (header) + N * (1 (ID) + 1 (Error) + length (data))
        expected_len = 11 + len(self.motor_ids) * (2 + length)
        response = self.serial.read(expected_len)

        if len(response) < 11 or response[8] != 0:
            return None # Error or timeout

        results = {}
        offset = 11
        for _ in self.motor_ids:
            if offset + 2 + length > len(response): break
            motor_id = response[offset]
            error = response[offset+1]
            if error == 0:
                data_start = offset + 2
                data_end = data_start + length
                results[motor_id] = int.from_bytes(response[data_start:data_end], 'little', signed=True)
            offset += (2 + length)
        return results

    def bulk_write(self, commands):
        params = bytearray()
        for motor_id, (address, length, data_bytes) in commands.items():
            params += motor_id.to_bytes(1, 'little')
            params += address.to_bytes(2, 'little')
            params += length.to_bytes(2, 'little')
            params += data_bytes
            
        packet = self._create_packet(self.BROADCAST_ID, self.INST_BULK_WRITE, params)
        self.serial.write(packet) # Fire-and-forget


# =============================================================================
#  2. HIGH-LEVEL ROBOT CONTROLLER (Your class, now using the I/O class)
# =============================================================================
class ServoControllerBD5():
    def __init__(self, port, baudrate=1000000):
        # Constants and robot definition
        self.PI = 3.14159265
        self.deg2rad = self.PI/180
        self.rad2deg = 180/self.PI
        self.rpm2rads = 0.10472
        
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

        # Value limit XM430-W350-T and XC430-W150-T
        self.MAX_POS = 4096
        self.MAX_ANG = 2*self.PI

        # Mem address for XM430 and XC430
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_RETURN_DELAY_TIME = 9
        self.ADDR_POSITION_P_GAIN = 84
        self.ADDR_POSITION_I_GAIN = 82
        self.ADDR_POSITION_D_GAIN = 80
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132

        # Len return mem value for XM430 and XC430
        self.LEN_TORQUE_ENABLE = 1
        self.LEN_RETURN_DELAY_TIME = 1
        self.LEN_POSITION_P_GAIN = 2
        self.LEN_POSITION_I_GAIN = 2
        self.LEN_POSITION_D_GAIN = 2
        self.LEN_GOAL_POSITION = 4
        self.LEN_PRESENT_POSITION = 4

        # Instantiate and open the low-level I/O controller
        self.dxl_io = DynamixelBulkControl(port, self.joint_ID_list, baudrate)
        if not self.dxl_io.open_port():
            raise Exception(f"Failed to open port {port}")
        print(f"Successfully opened port {port} at {baudrate} bps.")

    # --- Port Management ---
    def close_port(self):
        self.dxl_io.close_port()

    # --- Angle and Value Conversions ---
    def correctRotation(self, value, joint_list):
        return [v * c if v != 0.0 else 0.0 for v, c in zip(value, joint_list)]
    def dxlClamp(self, value):
        return [max(l[0], min(v, l[1])) for v, l in zip(value, self.joints_limit_list)]
    def dxl2position(self, value):
        return [(val - (self.MAX_POS/2)) * (self.MAX_ANG / (self.MAX_POS-1)) for val in value]
    def position2dxl(self, value):
        dxl_values = [int((self.MAX_POS/2) + (val / self.MAX_ANG) * (self.MAX_POS-1)) for val in value]
        return [max(0, min(self.MAX_POS - 1, val)) for val in dxl_values]

    # --- High-Level Robot Commands (Now using Bulk I/O) ---
    def ping(self):
        for id in self.joint_ID_list:
            success, model_num = self.dxl_io.ping(id)
            if success:
                print(f"Pinged ID: {id:03d} successfully! Model Number: {model_num}")
            else:
                print(f"Ping FAILED for ID: {id:03d}")
                return False
        return True

    def _bulk_write_all(self, address, length, value):
        """Helper to write the same value to the same address on all servos."""
        data_bytes = int(value).to_bytes(length, 'little')
        commands = {motor_id: (address, length, data_bytes) for motor_id in self.joint_ID_list}
        self.dxl_io.bulk_write(commands)

    def enable_torque(self):
        self._bulk_write_all(self.ADDR_TORQUE_ENABLE, self.LEN_TORQUE_ENABLE, 1)

    def disable_torque(self):
        self._bulk_write_all(self.ADDR_TORQUE_ENABLE, self.LEN_TORQUE_ENABLE, 0)
        
    def set_return_delay(self, value):
        print(f"Setting Return Delay Time for all servos to {value*2} us...")
        self._bulk_write_all(self.ADDR_RETURN_DELAY_TIME, self.LEN_RETURN_DELAY_TIME, value)

    def set_PID(self, pid):
        # This sends three separate bulk write packets, which is highly efficient.
        print(f"Setting PID for legs: P={pid[0]}, I={pid[1]}, D={pid[2]}")
        leg_ids = self.joint_ID_list[:10]
        # P Gain
        p_bytes = int(pid[0]).to_bytes(self.LEN_POSITION_P_GAIN, 'little')
        self.dxl_io.bulk_write({mid: (self.ADDR_POSITION_P_GAIN, self.LEN_POSITION_P_GAIN, p_bytes) for mid in leg_ids})
        # I Gain
        i_bytes = int(pid[1]).to_bytes(self.LEN_POSITION_I_GAIN, 'little')
        self.dxl_io.bulk_write({mid: (self.ADDR_POSITION_I_GAIN, self.LEN_POSITION_I_GAIN, i_bytes) for mid in leg_ids})
        # D Gain
        d_bytes = int(pid[2]).to_bytes(self.LEN_POSITION_D_GAIN, 'little')
        self.dxl_io.bulk_write({mid: (self.ADDR_POSITION_D_GAIN, self.LEN_POSITION_D_GAIN, d_bytes) for mid in leg_ids})

    def set_position(self, value_rad):
        value_rad_corrected = self.correctRotation(value_rad, self.joints_correction_list)
        ang_pos_dxl = self.position2dxl(value_rad_corrected)
        ang_pos_clamped = self.dxlClamp(ang_pos_dxl)
        
        commands = {}
        for i, motor_id in enumerate(self.joint_ID_list):
            pos_bytes = ang_pos_clamped[i].to_bytes(self.LEN_GOAL_POSITION, 'little', signed=True)
            commands[motor_id] = (self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION, pos_bytes)
        
        self.dxl_io.bulk_write(commands)

    def get_position(self):
        dxl_positions_dict = self.dxl_io.sync_read(self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
        
        if dxl_positions_dict is None or len(dxl_positions_dict) != len(self.joint_ID_list):
            print("[READ_ERROR] Failed to get positions for all servos.")
            return [], False
        
        # Order the results correctly
        ordered_dxl_pos = [dxl_positions_dict[mid] for mid in self.joint_ID_list]
        
        positions_rad = self.dxl2position(ordered_dxl_pos)
        positions_rad_corrected = self.correctRotation(positions_rad, self.joints_correction_list)
        
        return positions_rad_corrected, True


# =============================================================================
#  3. MAIN EXECUTION BLOCK (Largely unchanged, but with new class)
# =============================================================================
if __name__=='__main__':   
    # Dxl param
    port = "/dev/ttyUSB0"
    baudrate = 1000000

    try:
        # Init the new, integrated servos class
        BDX = ServoControllerBD5(port=port, baudrate=baudrate)
        
        # ping all servos 
        if BDX.ping():
            print("All servos ready!")
        else:
            BDX.close_port()
            raise Exception("Error pinging servos. Check IDs, power, or connections.")
        
        # Time param
        command_freq = 20
        ctrl_freq = 50
        ctrl_dt = 1.0 / ctrl_freq

        # Init gamepad
        controller = Gamepad(command_freq=command_freq, vel_range_x=[-0.6, 0.6], vel_range_y=[-0.6, 0.6], vel_range_rot=[-1.0, 1.0], head_range=[-0.5236, 0.5236], deadzone=0.02)

        # Default angles setup
        default_angles_leg = [0.0, -0.052, 0.872, 1.745, 0.872, 0.0, -0.052, 0.872, 1.745, 0.872]
        default_angles_head = [0.530, -0.530]
        default_angles_full = default_angles_leg + default_angles_head
        
        print("\nWaiting for [CIRCLE] button to activate robot...")
        while True:
            _, _, _, _, C_pressed, _ = controller.get_last_command()
            if C_pressed:
                print("BD-5 ACTIVATED!")
                break
            time.sleep(0.1)

        # Activate + Set default configuration
        BDX.set_return_delay(value=50) # Set delay to 2us for max speed
        BDX.set_PID(pid=[800, 0, 0])
        BDX.enable_torque()
        time.sleep(0.1)
        BDX.set_position(default_angles_full)
        time.sleep(0.5) # Give time to reach start position
        
        print("Starting 50Hz control loop... Press [X] to stop.")
        start_t = time.time()
        i = 0
        while True:
            t_loop_start = time.perf_counter()

            # Get command from gamepad
            last_state, head_t, S_pressed, T_pressed, C_pressed, X_pressed = controller.get_last_command()
            if X_pressed:
                print("Kill switch pressed!")
                break

            # --- CORE CONTROL LOOP ---
            # 1. READ all servo positions (1 blocking transaction)
            pos, state = BDX.get_position()
            print(pos)
            if not state:
                continue # Skip loop if read failed
            
            # 3. WRITE all servo positions (1 non-blocking transaction)
            BDX.set_position(default_angles_full)
            # --- END OF CORE CONTROL ---
            
            # Time control to maintain 50Hz
            took = time.perf_counter() - t_loop_start
            if (ctrl_dt - took) < 0:
                print(f"Control budget exceeded by {abs(ctrl_dt - took)*1000:.2f} ms")
            time.sleep(max(0, ctrl_dt - took))
            
            i+=1
        
        total_time = time.time() - start_t
        print(f"\nRan {i} loops in {total_time:.2f} seconds. Average freq: {i/total_time:.2f} Hz")

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
    
    finally:
        print("Disabling torque and closing port...")
        try:
            # Gracefully shutdown
            BDX.disable_torque()
            time.sleep(0.5)
            BDX.close_port()
            print("Port closed.")
        except NameError:
            # This happens if BDX failed to initialize
            print("Controller was not initialized. Nothing to close.")
        except Exception as e:
            print(f"Error during shutdown: {e}")

    
