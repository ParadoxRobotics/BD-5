

import numpy as np
import matplotlib.pyplot as plt

Amplitude=0.04 
NbStep=2
Sample=250
LegStride=0.04 
LegHeight=0.03 
InitLegLength=0.15

data_y = []
data_rx = []
data_rz = []
data_lx = []
data_lz = [] 

for j in range(0, NbStep):
    for i in range(0, Sample):
        # Lateral trajectory
        hip_y = Amplitude * np.sin(np.pi * 2 * i / Sample) # lateral phase
        data_y.append(hip_y)
        # Stride trajectory
        right_x = LegStride * np.sin(np.pi * 2 * i / Sample - np.pi / 2) # phase (-90°)
        left_x = LegStride * np.sin(np.pi * 2 * i / Sample + np.pi / 2) # phase (+90°)
        data_rx.append(right_x)
        data_lx.append(left_x)
        # Foot lifting trajectory
        right_z = InitLegLength - LegHeight * np.sin(np.pi * 2 * i / Sample) # phase (-180°)
        left_z = InitLegLength + LegHeight * np.sin(np.pi * 2 * i / Sample) # phase (+180°)
        # Apply z position only for foot lifting
        if (right_z > InitLegLength):
            right_z = InitLegLength
        data_rz.append(right_z)
        if (left_z > InitLegLength):
            left_z = InitLegLength
        data_lz.append(left_z)

        
fig, axs = plt.subplots(5)
fig.suptitle('gait trajectory')
axs[0].plot(data_y, [*range(0, len(data_y))])
axs[1].plot(data_rx, [*range(0, len(data_rx))])
axs[2].plot(data_lx, [*range(0, len(data_lx))])
axs[3].plot(data_rz, [*range(0, len(data_rz))])
axs[4].plot(data_lz, [*range(0, len(data_lz))])
plt.tight_layout()
plt.show()