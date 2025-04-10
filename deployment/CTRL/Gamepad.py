import pygame
from threading import Thread
from queue import Queue
import time
import numpy as np

class Gamepad:
    def __init__(self, command_freq, vel_range_x, vel_range_y, vel_range_rot, deadzone):
        self.command_freq = command_freq
        self.vel_range_x = vel_range_x
        self.vel_range_y = vel_range_y
        self.vel_range_rot = vel_range_rot
        self.deadzone = deadzone

        self.last_commands = [0.0, 0.0, 0.0]
        self.last_left_trigger = 0.0
        self.last_right_trigger = 0.0

        pygame.init()
        self.p1 = pygame.joystick.Joystick(0)
        self.p1.init()
        print(f"Loaded joystick with {self.p1.get_numaxes()} axes.")

        self.cmd_queue = Queue(maxsize=1)
        Thread(target=self.commands_worker, daemon=True).start()

    def commands_worker(self):
        while True:
            self.cmd_queue.put(self.get_commands())
            time.sleep(1 / self.command_freq)

    def get_commands(self):
        A_pressed = False
        X_pressed = False
        B_pressed = False
        Y_pressed = False

        last_commands = self.last_commands

        l_x = -1 * self.p1.get_axis(0)
        l_y = -1 * self.p1.get_axis(1)
        r_x = -1 * self.p1.get_axis(3)
        h_t = -1 * self.p1.get_axis(4)

        lin_vel_y = l_x
        lin_vel_x = l_y
        ang_vel = r_x
        head_t = h_t

        if lin_vel_x >= 0:
            lin_vel_x *= np.abs(self.vel_range_x[1])
        else:
            lin_vel_x *= np.abs(self.vel_range_x[0])

        if lin_vel_y >= 0:
            lin_vel_y *= np.abs(self.vel_range_y[1])
        else:
            lin_vel_y *= np.abs(self.vel_range_y[0])

        if ang_vel >= 0:
            ang_vel *= np.abs(self.vel_range_rot[1])
        else:
            ang_vel *= np.abs(self.vel_range_rot[0])

        if head_t >= 0:
            head_t *= np.abs(1.0)
        else:
            head_t *= np.abs(-1.0)


        if abs(lin_vel_x) < self.deadzone:
            lin_vel_x = 0.0
        if abs(lin_vel_y) < self.deadzone:
            lin_vel_y = 0.0
        if abs(ang_vel) < self.deadzone:
            ang_vel = 0.0
        if abs(head_t) < self.deadzone:
            head_t = 0.0

        last_commands[0] = lin_vel_x
        last_commands[1] = lin_vel_y
        last_commands[2] = ang_vel

        for event in pygame.event.get():
            if self.p1.get_button(0):  # A button
                A_pressed = True

            if self.p1.get_button(3):  # X button
                X_pressed = True

            if self.p1.get_button(1):  # B button
                B_pressed = True

            if self.p1.get_button(2):  # Y button
                Y_pressed = True

        pygame.event.pump()  # process event queue

        return np.around(last_commands, 3), head_t, A_pressed, X_pressed, B_pressed, Y_pressed

    def get_last_command(self):
        A_pressed = False
        X_pressed = False
        B_pressed = False
        Y_pressed = False
        try:
            self.last_commands, head_tilt, A_pressed, X_pressed, B_pressed, Y_pressed = self.cmd_queue.get(False)  # non blocking
        except Exception:
            pass
        return self.last_commands, head_tilt, A_pressed, X_pressed, B_pressed, Y_pressed


if __name__ == "__main__":
    controller = Gamepad(command_freq=20, vel_range_x=[-0.6, 0.6], vel_range_y=[-0.6, 0.6], vel_range_rot=[-1.0, 1.0], deadzone=0.03)

    while True:
        print(controller.get_last_command())
        time.sleep(0.05)