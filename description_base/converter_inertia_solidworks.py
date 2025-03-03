# 原始字符串
raw_string = """
所选零部件 的质量属性
     坐标系： BCS

质量 = 5439.19 克

体积 = 2525851.14 立方毫米

表面积 = 979087.02  平方毫米

重心 : ( 毫米 )
	X = 0.00
	Y = 1.10
	Z = 7.33

惯性主轴和惯性主力矩: ( 克 *  平方毫米 )
由重心决定。
	 Ix = ( 0.00, -1.00, -0.01)   	Px = 45020646.47
	 Iy = ( 1.00,  0.00,  0.00)   	Py = 48410923.95
	 Iz = ( 0.00, -0.01,  1.00)   	Pz = 80241299.66

惯性张量: ( 克 *  平方毫米 )
由重心决定，并且对齐输出的坐标系。 （使用正张量记数法。）
	Lxx = 48410923.95	Lxy = 1.36	Lxz = -0.97
	Lyx = 1.36	Lyy = 45021610.36	Lyz = 184249.38
	Lzx = -0.97	Lzy = 184249.38	Lzz = 80240335.77

惯性张量: ( 克 *  平方毫米 )
由输出座标系决定。 （使用正张量记数法。）
	Ixx = 48709740.16	Ixy = 0.92	Ixz = -3.91
	Iyx = 0.92	Iyy = 45313794.64	Iyz = 228269.22
	Izx = -3.91	Izy = 228269.22	Izz = 80246967.70
"""

# 提取数值并除以10^9
mass = str(float(raw_string.split('质量 = ')[1].split()[0]) / 1e3)
cog_x = str(float(raw_string.split('X = ')[1].split()[0]) / 1e3)
cog_y = str(float(raw_string.split('Y = ')[1].split()[0]) / 1e3)
cog_z = str(float(raw_string.split('Z = ')[1].split()[0]) / 1e3)
ixx = str(float(raw_string.split('Lxx = ')[1].split()[0]) / 1e9)
ixy = str(-float(raw_string.split('Lxy = ')[1].split()[0]) / 1e9)
ixz = str(-float(raw_string.split('Lxz = ')[1].split()[0]) / 1e9)
iyy = str(float(raw_string.split('Lyy = ')[1].split()[0]) / 1e9)
iyz = str(-float(raw_string.split('Lyz = ')[1].split()[0]) / 1e9)
izz = str(float(raw_string.split('Lzz = ')[1].split()[0]) / 1e9)

# 输出格式
output_string = f"""
<inertial>
  <origin
    xyz="{cog_x} {cog_y} {cog_z}"
    rpy="0 0 0" />
  <mass
    value="{mass}" />
  <inertia
    ixx="{ixx}"
    ixy="{ixy}"
    ixz="{ixz}"
    iyy="{iyy}"
    iyz="{iyz}"
    izz="{izz}" />
</inertial>
"""

print(output_string)