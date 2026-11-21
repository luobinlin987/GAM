import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# Geometry
print("Geometry")
start_geometry = time.perf_counter()
from functions import arc_points, charge_point, forward_mapping_solution, backward_mapping_solution, forward_mapping, backward_mapping, backward_mapping_d, backward_mapping_d2, inside_C1, inside_C2, inside_C3, inside_C4, inside_C0
N = 0
# Geometry.C0
# Forward assigment factor
RjC0 = 2.5
# Backward assigment factor
rjc0 = 2.0
# C0, square, size 20*20, local coordinate origin in global coordinate system (0,0)
# Sharp corner should be smoothened into rounded corner
# Illustration
# arc —————————————— arc
# |                    |
# |                    |
# |        (0,0)       |
# |                    |
# |                    |
# arc —————————————— arc
z_0c = 0.
# Sample point distribution should be clock-wise
# arc radius 0.5
C0r = 0.5
a0 = 5.6
b0 = 5.6
# Straight segment length 20-2*0.5
x0C0local, x1C0local = -(a0+C0r), (a0+C0r)
y0C0local, y1C0local = -(b0+C0r), (b0+C0r)
# Straight segment sample point quantity
C0local_horizontal_num = int(40*a0)
C0local_vertical_num = int(40*b0)
# Straight segment sample points
xC0local_straight = np.linspace(x0C0local+C0r, x1C0local-C0r, C0local_horizontal_num)
yC0local_straight = np.linspace(y0C0local+C0r, y1C0local-C0r, C0local_vertical_num)
# Straight line segments, four segments
C0straight_edge_top = np.column_stack([xC0local_straight[1:-1], np.full_like(xC0local_straight[1:-1],y1C0local)])
C0straight_edge_bottom = np.column_stack([xC0local_straight[1:-1], np.full_like(xC0local_straight[1:-1],y0C0local)])
C0straight_edge_left = np.column_stack([np.full_like(yC0local_straight[1:-1],x0C0local),yC0local_straight[1:-1]])
C0straight_edge_right = np.column_stack([np.full_like(yC0local_straight[1:-1],x1C0local),yC0local_straight[1:-1]])
# Arc segment sample point quantity
C0local_arc_num = 30
# Arc center in local coordinate system
C0local_arc_xc_above_left = x0C0local+C0r
C0local_arc_yc_above_left = y1C0local-C0r
C0local_arc_xc_bottom_left = x0C0local+C0r
C0local_arc_yc_bottom_left = y0C0local+C0r
C0local_arc_xc_above_right = x1C0local-C0r
C0local_arc_yc_above_right = y1C0local-C0r
C0local_arc_xc_bottom_right = x1C0local-C0r
C0local_arc_yc_bottom_right = y0C0local+C0r
# Arc sample points, using function arc_points using function arc_points
C0local_arc_above_left = arc_points(C0local_arc_xc_above_left, C0local_arc_yc_above_left, C0r, np.pi/2, np.pi, C0local_arc_num)
C0local_arc_bottom_left = arc_points(C0local_arc_xc_bottom_left, C0local_arc_yc_bottom_left, C0r, np.pi, np.pi/2*3, C0local_arc_num)
C0local_arc_above_right = arc_points(C0local_arc_xc_above_right, C0local_arc_yc_above_right, C0r, 0., np.pi/2, C0local_arc_num)
C0local_arc_bottom_right = arc_points(C0local_arc_xc_bottom_right, C0local_arc_yc_bottom_right, C0r, np.pi/2*3, np.pi*2, C0local_arc_num)
# Collocation point distribution along C0 periphery, anti-clockwise
C0local_periphery = np.vstack([C0straight_edge_top[::-1], C0local_arc_above_left[::], C0straight_edge_left[::-1], C0local_arc_bottom_left[::], C0straight_edge_bottom[::], C0local_arc_bottom_right[::], C0straight_edge_right[::], C0local_arc_above_right[::]])
C0local_periphery_z = C0local_periphery[:,0] + 1j * C0local_periphery[:,1]
# Charge point distribution along C0 periphery, also ant-clockwise
C0local_periphery_X = charge_point(C0local_periphery_z, RjC0)
# Forward mapping solution for C0
QC0, Gamma_0 = forward_mapping_solution(C0local_periphery_z, C0local_periphery_X, 0, N, z_0c)
# Compute forward mapping point along C0
forward_mapping_along_C0 = forward_mapping(C0local_periphery_z, C0local_periphery_X, Gamma_0, QC0, 0, z_0c, N)
forward_mapping_along_C0x = np.real(forward_mapping_along_C0)
forward_mapping_along_C0y = np.imag(forward_mapping_along_C0)
# Forward conformal mapping for C0 finishes
# Backward conformal mapping for C0 starts
c0local_periphery_zeta = forward_mapping_along_C0
# Compute charge points along mapped unit-radius periphery in mapping plane
c0local_periphery_x = charge_point(c0local_periphery_zeta, rjc0)
# Backward mapping solution for c0
qc0 = backward_mapping_solution(C0local_periphery_z, c0local_periphery_zeta, c0local_periphery_x, 0, N)
# Geometry.C1
# Forward assigment factor
RjC1 = 2.0
# Backward assigment factor
rjc1 = 2.0
# Reference point should be located at center of the local coordinate system due to symmetry about both x and y axes
z_1c = 0. + 1j * 0.
# Geometry of C1 is an ellipse with radius 3
theta_num = 360
r1 = 3
theta11 = np.linspace(2.*np.pi, 0, theta_num, endpoint = False)
C1local_periphery_x = r1 * np.cos(theta11)
C1local_periphery_y = r1 * np.sin(theta11)
C1local_periphery = np.column_stack([C1local_periphery_x,C1local_periphery_y])
C1local_periphery_z = C1local_periphery_x + 1j * C1local_periphery_y
# Forward conformal mapping
C1local_periphery_X = charge_point(C1local_periphery_z, RjC1)
# Forward mapping solution for C1
QC1, Gamma_1 = forward_mapping_solution(C1local_periphery_z, C1local_periphery_X, 1, N, z_1c)
# Compute forward mapping point along C1
forward_mapping_along_C1 = forward_mapping(C1local_periphery_z, C1local_periphery_X, Gamma_1, QC1, 1, z_1c, N)
forward_mapping_along_C1x = np.real(forward_mapping_along_C1)
forward_mapping_along_C1y = np.imag(forward_mapping_along_C1)
# Forward conformal mapping for C1 finishes
# Backward conformal mapping for C1 starts
c1local_periphery_zeta = forward_mapping_along_C1
# Compute charge points along mapped unit-radius periphery in mapping plane
c1local_periphery_x = charge_point(c1local_periphery_zeta, rjc1)
# Backward mapping solution for c1
qc1 = backward_mapping_solution(C1local_periphery_z, c1local_periphery_zeta, c1local_periphery_x, 1, N)
# Geometry.C2
# Forward assigment factor
RjC2 = 2.0
# Backward assigment factor
rjc2 = 2.0
# C2, square, size 6*6, local coordinate origin in global coordinate system (-5,0)
# Sharp corner should be smoothened into rounded corner
# Illustration
# arc —————————— arc
# |               |
# |               |
# |     (-5,0)    |
# |               |
# |               |
# arc —————————— arc
# Reference point should be located at center of the local coordinate system due to symmetry about both x and y axes
z_2c = 0. + 1j * 0.
# Sample point distribution should be clock-wise
# arc radius 0.1
C2r = 0.1
# Straight segment length 5.8
x0C2local, x1C2local = -3.0, 3.0
y0C2local, y1C2local = -3.0, 3.0
# Straight segment sample point quantity
C2local_straight_num = 100
# Straight segment sample points
xC2local_straight = np.linspace(x0C2local+C2r, x1C2local-C2r, C2local_straight_num)
yC2local_straight = np.linspace(y0C2local+C2r, y1C2local-C2r, C2local_straight_num)
# Straight line segments, four segments
C2straight_edge_top = np.column_stack([xC2local_straight[1:-1], np.full_like(xC2local_straight[1:-1],y1C2local)])
C2straight_edge_bottom = np.column_stack([xC2local_straight[1:-1], np.full_like(xC2local_straight[1:-1],y0C2local)])
C2straight_edge_left = np.column_stack([np.full_like(yC2local_straight[1:-1],x0C2local),yC2local_straight[1:-1]])
C2straight_edge_right = np.column_stack([np.full_like(yC2local_straight[1:-1],x1C2local),yC2local_straight[1:-1]])
# Arc segment sample point quantity
C2local_arc_num = 30
# Arc center in local coordinate system
C2local_arc_xc_above_left = x0C2local+C2r
C2local_arc_yc_above_left = y1C2local-C2r
C2local_arc_xc_bottom_left = x0C2local+C2r
C2local_arc_yc_bottom_left = y0C2local+C2r
C2local_arc_xc_above_right = x1C2local-C2r
C2local_arc_yc_above_right = y1C2local-C2r
C2local_arc_xc_bottom_right = x1C2local-C2r
C2local_arc_yc_bottom_right = y0C2local+C2r
# Arc sample points, using function arc_points using function arc_points
C2local_arc_above_left = arc_points(C2local_arc_xc_above_left, C2local_arc_yc_above_left, C2r, np.pi/2, np.pi, C2local_arc_num)
C2local_arc_bottom_left = arc_points(C2local_arc_xc_bottom_left, C2local_arc_yc_bottom_left, C2r, np.pi, np.pi/2*3, C2local_arc_num)
C2local_arc_above_right = arc_points(C2local_arc_xc_above_right, C2local_arc_yc_above_right, C2r, 0., np.pi/2, C2local_arc_num)
C2local_arc_bottom_right = arc_points(C2local_arc_xc_bottom_right, C2local_arc_yc_bottom_right, C2r, np.pi/2*3, np.pi*2, C2local_arc_num)
# Collocation point distribution along C2 periphery, clock-wise
C2local_periphery = np.vstack([C2straight_edge_top, C2local_arc_above_right[::-1], C2straight_edge_right[::-1], C2local_arc_bottom_right[::-1], C2straight_edge_bottom[::-1], C2local_arc_bottom_left[::-1], C2straight_edge_left, C2local_arc_above_left[::-1]])
C2local_periphery_z = C2local_periphery[:,0] + 1j * C2local_periphery[:,1]
# Charge point distribution along C2 periphery, also clock-wise
C2local_periphery_X = charge_point(C2local_periphery_z, RjC2)
# Forward mapping solution for C2
QC2, Gamma_2 = forward_mapping_solution(C2local_periphery_z, C2local_periphery_X, 2, N, z_2c)
# Compute forward mapping point along C2
forward_mapping_along_C2 = forward_mapping(C2local_periphery_z, C2local_periphery_X, Gamma_2, QC2, 2, z_2c, N)
forward_mapping_along_C2x = np.real(forward_mapping_along_C2)
forward_mapping_along_C2y = np.imag(forward_mapping_along_C2)
# Forward conformal mapping for C2 finishes
# Backward conformal mapping for C2 starts
c2local_periphery_zeta = forward_mapping_along_C2
# Compute charge points along mapped unit-radius periphery in mapping plane
c2local_periphery_x = charge_point(c2local_periphery_zeta, rjc2)
# Backward mapping solution for c2
qc2 = backward_mapping_solution(C2local_periphery_z, c2local_periphery_zeta, c2local_periphery_x, 2, N)
# Geometry.C3
# Forward assigment factor
RjC3 = 2.0
# Backward assigment factor
rjc3 = 2.0
# Reference point should be located at center of the local coordinate system due to symmetry about both x and y axes
z_3c = 0. + 1j * 0.
# Geometry of C3 is an ellipse with x radius 3 and y radius 2
a3 = 3.
b3 = 2
theta33 = np.linspace(2.*np.pi, 0, theta_num, endpoint = False)
C3local_periphery_x = a3 * np.cos(theta33)
C3local_periphery_y = b3 * np.sin(theta33)
C3local_periphery = np.column_stack([C3local_periphery_x,C3local_periphery_y])
C3local_periphery_z = C3local_periphery_x + 1j * C3local_periphery_y
# Forward conformal mapping
C3local_periphery_X = charge_point(C3local_periphery_z, RjC3)
# Forward mapping solution for C3
QC3, Gamma_3 = forward_mapping_solution(C3local_periphery_z, C3local_periphery_X, 3, N, z_3c)
# Compute forward mapping point along C3
forward_mapping_along_C3 = forward_mapping(C3local_periphery_z, C3local_periphery_X, Gamma_3, QC3, 3, z_3c, N)
forward_mapping_along_C3x = np.real(forward_mapping_along_C3)
forward_mapping_along_C3y = np.imag(forward_mapping_along_C3)
# Forward conformal mapping for C3 finishes
# Backward conformal mapping for C3 starts
c3local_periphery_zeta = forward_mapping_along_C3
# Compute charge points along mapped unit-radius periphery in mapping plane
c3local_periphery_x = charge_point(c3local_periphery_zeta, rjc3)
# Backward mapping solution for c3
qc3 = backward_mapping_solution(C3local_periphery_z, c3local_periphery_zeta, c3local_periphery_x, 3, N)
# Gometry.C4
# Forward assigment factor
RjC4 = 2.
# Backward assigment factor
rjc4 = 2.
# Reference point should be located at as much near center of the local coordinate system as possible due to sole symmetry about both x axis
z_4c = 0. + 1j * 0.5
# C4 is composed of an upper half circle of radius 3 and a lower ellipse of x radius 3 and y radius 2
a4 = 3.
b4up = 3.
b4lw = 2.
theta_up = np.linspace(np.pi, 0., theta_num//2, endpoint = False)
theta_lw = np.linspace(2.*np.pi, np.pi, theta_num//2, endpoint = False)
C4local_periphery_x_up = a4 * np.cos(theta_up)
C4local_periphery_y_up = b4up * np.sin(theta_up)
C4local_periphery_x_lw = a4 * np.cos(theta_lw)
C4local_periphery_y_lw = b4lw * np.sin(theta_lw)
C4local_periphery_x = np.concatenate([C4local_periphery_x_up, C4local_periphery_x_lw])
C4local_periphery_y = np.concatenate([C4local_periphery_y_up, C4local_periphery_y_lw])
C4local_periphery = np.column_stack([C4local_periphery_x,C4local_periphery_y])
C4local_periphery_z = C4local_periphery_x + 1j * C4local_periphery_y
# Forward conformal mapping
C4local_periphery_X = charge_point(C4local_periphery_z, RjC4)
# Forward mapping solution for C4
QC4, Gamma_4 = forward_mapping_solution(C4local_periphery_z, C4local_periphery_X, 4, N, z_4c)
# Compute forward mapping point along C4
forward_mapping_along_C4 = forward_mapping(C4local_periphery_z, C4local_periphery_X, Gamma_4, QC4, 4, z_4c, N)
forward_mapping_along_C4x = np.real(forward_mapping_along_C4)
forward_mapping_along_C4y = np.imag(forward_mapping_along_C4)
# Forward conformal mapping for C4 finishes
# Backward conformal mapping for C4 starts
c4local_periphery_zeta = forward_mapping_along_C4
# Compute charge points along mapped unit-radius periphery in mapping plane
c4local_periphery_x = charge_point(c4local_periphery_zeta, rjc4)
# Backward mapping solution for c4
qc4 = backward_mapping_solution(C4local_periphery_z, c4local_periphery_zeta, c4local_periphery_x, 4, N)
end_geometry = time.perf_counter()
time_geometry = end_geometry - start_geometry
print(f"  Conformal mapping, time = {time_geometry:.4f} seconds")
# ----------------------------------------------------------
print("Solution preparation")
start_prepare = time.perf_counter()
zO = [0, -(a0+C0r)/2+1j*(a0+C0r)/2, (a0+C0r)/2+1j*(a0+C0r)/2, -(a0+C0r)/2-1j*(a0+C0r)/2, (a0+C0r)/2-1j*(a0+C0r)/2]
X = [C0local_periphery_X, C1local_periphery_X, C2local_periphery_X, C3local_periphery_X, C4local_periphery_X]
x = [c0local_periphery_x, c1local_periphery_x, c2local_periphery_x, c3local_periphery_x, c4local_periphery_x]
Gamma = [Gamma_0, Gamma_1, Gamma_2, Gamma_3, Gamma_4]
QC = [QC0, QC1, QC2, QC3, QC4]
zc = [z_0c, z_1c, z_2c, z_3c, z_4c]
qc = [qc0, qc1, qc2, qc3, qc4]
C0local_periphery_z_shift = np.roll(C0local_periphery_z, -1)
C1local_periphery_z_shift = np.roll(C1local_periphery_z, -1)
C2local_periphery_z_shift = np.roll(C2local_periphery_z, -1)
C3local_periphery_z_shift = np.roll(C3local_periphery_z, -1)
C4local_periphery_z_shift = np.roll(C4local_periphery_z, -1)
Cz0 = [C0local_periphery_z, C1local_periphery_z, C2local_periphery_z, C3local_periphery_z, C4local_periphery_z]
Cz0_shift = [C0local_periphery_z_shift, C1local_periphery_z_shift, C2local_periphery_z_shift, C3local_periphery_z_shift, C4local_periphery_z_shift]
czeta0 = [c0local_periphery_zeta, c1local_periphery_zeta, c2local_periphery_zeta, c3local_periphery_zeta, c4local_periphery_zeta]
Nj =  [4,4,4,4,4]
zlength = 0
for j in range(5):
		zlength += Nj[j]*len(Cz0[j])
Cz = np.zeros((zlength), dtype = np.complex128)
startj = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		for J in range(Nj[j]):
				Cz[startj+J:endj:Nj[j]] = J/Nj[j]*Cz0_shift[j]+(Nj[j]-J)/Nj[j]*Cz0[j]
		startj += lengthj
EE = 1.
Lr = 1.
Lt = 1.
mu = 0.3
Km = 1
KG = EE/(1+mu)
kappa = 3-4*mu
lambdajr = [0,0.25,0.5,0.75,1]
lambdajt = [1,0.75,0.5,0.25,0]
gammaj1 = [0.4, 0.4, 0.3, 0.3, 0.3]
gammaj2 = [0.4, 0.4, 0.3, 0.3, 0.3]
czeta = np.zeros((zlength), dtype = np.complex128)
startj = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		czeta[startj:endj] = forward_mapping(Cz[startj:endj], X[j], Gamma[j], QC[j], j, zc[j], N)
		startj += lengthj
Klength = 0
for J in range(5):
		Klength += len(X[J])
etaJ1k = np.zeros((Klength), dtype = np.complex128)
etaJ2k = np.zeros((Klength), dtype = np.complex128)
startJ = 0
for J in range(5):
		lengthJ = len(X[J])
		endJ = startJ+lengthJ
		etaJ1k[startJ:endJ] = charge_point(czeta0[J], gammaj1[J])
		etaJ2k[startJ:endJ] = charge_point(czeta0[J], gammaj2[J])
		startJ += lengthJ
czetajJ = np.zeros((zlength,5), dtype = np.complex128)
zdjJ = np.zeros((zlength,5), dtype = np.complex128)
zd2jJ = np.zeros((zlength,5), dtype = np.complex128)
startj = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		for J in range(5):
				czetajJ[startj:endj,J] = forward_mapping(Cz[startj:endj]+zO[j]-zO[J], X[J], Gamma[J], QC[J], J, zc[J], N)
				zdjJ[startj:endj,J] = backward_mapping_d(czetajJ[startj:endj,J], x[J], qc[J], J, N)
				zd2jJ[startj:endj,J] = backward_mapping_d2(czetajJ[startj:endj,J], x[J], qc[J], J, N)
		startj += lengthj
Pjj = np.zeros((zlength), dtype = np.complex128)
Qjj = np.zeros((zlength), dtype = np.complex128)
startj = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		Pjj[startj:endj] = czetajJ[startj:endj,j]*zdjJ[startj:endj,j]/np.abs(czetajJ[startj:endj,j]*zdjJ[startj:endj,j])
		Qjj[startj:endj] = czetajJ[startj:endj,j]*zdjJ[startj:endj,j]/np.conj(czetajJ[startj:endj,j]*zdjJ[startj:endj,j])
		startj += lengthj
Cz_p = np.zeros((zlength,Klength), dtype = np.complex128)
startj = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		Cz_p[startj:endj,:] = (Cz[startj:endj]+zO[j])[:,None]
		startj += lengthj
czetajJ_p = np.zeros((zlength,Klength), dtype = np.complex128)
zdjJ_p = np.zeros((zlength,Klength), dtype = np.complex128)
zd2jJ_p = np.zeros((zlength,Klength), dtype = np.complex128)
startJ = 0
for J in range(5):
		lengthJ = len(X[J])
		endJ = startJ+lengthJ
		czetajJ_p[:,startJ:endJ] = czetajJ[:,J][:,None]
		zdjJ_p[:,startJ:endJ] = zdjJ[:,J][:,None]
		zd2jJ_p[:,startJ:endJ] = zd2jJ[:,J][:,None]
		startJ += lengthJ
etaJ1k_p = np.zeros((zlength,Klength), dtype = np.complex128)
etaJ2k_p = np.zeros((zlength,Klength), dtype = np.complex128)
startJ = 0
for J in range(5):
		lengthJ = len(X[J])
		endJ = startJ+lengthJ
		etaJ1k_p[:,startJ:endJ] = etaJ1k[startJ:endJ]
		etaJ2k_p[:,startJ:endJ] = etaJ2k[startJ:endJ]
		startJ += lengthJ
Pjj_p = np.zeros((zlength,Klength), dtype = np.complex128)
Qjj_p = np.zeros((zlength,Klength), dtype = np.complex128)
Pjj_p[:,:] = Pjj[:,None]
Qjj_p[:,:] = Qjj[:,None]
g1jJk = np.zeros((zlength,Klength), dtype = np.complex128)
g2jJk = np.zeros((zlength,Klength), dtype = np.complex128)
g3jJk = np.zeros((zlength,Klength), dtype = np.complex128)
g4jJk = np.zeros((zlength,Klength), dtype = np.complex128)
g5jJk = np.zeros((zlength,Klength), dtype = np.complex128)
g6jJk = np.zeros((zlength,Klength), dtype = np.complex128)
startj = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		startJ = 0
		for J in range(5):
				lengthJ = len(X[J])
				endJ = startJ+lengthJ
				g1jJk[startj:endj,startJ:endJ] = -1./(zdjJ_p[startj:endj,startJ:endJ]*(czetajJ_p[startj:endj,startJ:endJ]-etaJ1k_p[startj:endj,startJ:endJ])**2)
				g2jJk[startj:endj,startJ:endJ] = -2.*Cz_p[startj:endj,startJ:endJ]/((np.conj(zdjJ_p[startj:endj,startJ:endJ]))**(2)*(np.conj(czetajJ_p[startj:endj,startJ:endJ]-etaJ1k_p[startj:endj,startJ:endJ]))**3) - Cz_p[startj:endj,startJ:endJ]*np.conj(zd2jJ_p[startj:endj,startJ:endJ])/(np.conj(zdjJ_p[startj:endj,startJ:endJ])**3*(np.conj(czetajJ_p[startj:endj,startJ:endJ]-etaJ1k_p[startj:endj,startJ:endJ]))**2)
				g3jJk[startj:endj,startJ:endJ] = 1./(np.conj(zdjJ_p[startj:endj,startJ:endJ])*np.conj(czetajJ_p[startj:endj,startJ:endJ]-etaJ2k_p[startj:endj,startJ:endJ])**2)
				g4jJk[startj:endj,startJ:endJ] = kappa/(czetajJ_p[startj:endj,startJ:endJ]-etaJ1k_p[startj:endj,startJ:endJ])
				g5jJk[startj:endj,startJ:endJ] = Cz_p[startj:endj,startJ:endJ]/(np.conj(zdjJ_p[startj:endj,startJ:endJ])*(np.conj(czetajJ_p[startj:endj,startJ:endJ]-etaJ1k_p[startj:endj,startJ:endJ]))**(2))
				g6jJk[startj:endj,startJ:endJ] = -1./(np.conj(czetajJ_p[startj:endj,startJ:endJ]-etaJ2k_p[startj:endj,startJ:endJ]))
				startJ += lengthJ
		startj += lengthj
f1jJk = np.zeros((zlength,Klength), dtype = np.complex128)
f2jJk = np.zeros((zlength,Klength), dtype = np.complex128)
f3jJk = np.zeros((zlength,Klength), dtype = np.complex128)
f4jJk = np.zeros((zlength,Klength), dtype = np.complex128)
startj = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		startJ = 0
		for J in range(5):
				lengthJ = len(X[J])
				endJ = startJ+lengthJ
				f1jJk[startj:endj,startJ:endJ] = (1-lambdajr[j])*Lr*KG**(-1)*(np.conj(g4jJk[startj:endj,startJ:endJ])*Pjj_p[startj:endj,startJ:endJ]+g5jJk[startj:endj,startJ:endJ]*np.conj(Pjj_p[startj:endj,startJ:endJ]))+lambdajr[j]*Km*(2*np.conj(g1jJk[startj:endj,startJ:endJ])+g2jJk[startj:endj,startJ:endJ]*np.conj(Qjj_p[startj:endj,startJ:endJ]))
				f2jJk[startj:endj,startJ:endJ] = (1-lambdajr[j])*Lr*KG**(-1)*g6jJk[startj:endj,startJ:endJ]*np.conj(Pjj_p[startj:endj,startJ:endJ])+lambdajr[j]*Km*g3jJk[startj:endj,startJ:endJ]*np.conj(Qjj_p[startj:endj,startJ:endJ])
				f3jJk[startj:endj,startJ:endJ] = (1-lambdajt[j])*Lt*KG**(-1)*(g4jJk[startj:endj,startJ:endJ]*np.conj(Pjj_p[startj:endj,startJ:endJ])-np.conj(g5jJk[startj:endj,startJ:endJ])*Pjj_p[startj:endj,startJ:endJ])-lambdajt[j]*Km*np.conj(g2jJk[startj:endj,startJ:endJ])*Qjj_p[startj:endj,startJ:endJ]
				f4jJk[startj:endj,startJ:endJ] = -(1-lambdajt[j])*Lt*KG**(-1)*np.conj(g6jJk[startj:endj,startJ:endJ])*Pjj_p[startj:endj,startJ:endJ]-lambdajt[j]*Km*np.conj(g3jJk[startj:endj,startJ:endJ])*Qjj_p[startj:endj,startJ:endJ]
				startJ += lengthJ
		startj += lengthj
del g1jJk, g2jJk, g3jJk, g4jJk, g5jJk, g6jJk
del Cz_p, czetajJ_p, zdjJ_p, zd2jJ_p, etaJ1k_p, etaJ2k_p, Pjj_p, Qjj_p
Gj = np.zeros((zlength), dtype = np.complex128)
Gj[:Nj[0]*len(Cz0[0])] = 1
Gj[Nj[0]*len(Cz0[0]):Nj[0]*len(Cz0[0])+Nj[1]*len(Cz0[1])] = 0
Gj[Nj[0]*len(Cz0[0])+Nj[1]*len(Cz0[1]):Nj[0]*len(Cz0[0])+Nj[1]*len(Cz0[1])+Nj[2]*len(Cz0[2])] = 0
Gj[Nj[0]*len(Cz0[0])+Nj[1]*len(Cz0[1])+Nj[2]*len(Cz0[2]):Nj[0]*len(Cz0[0])+Nj[1]*len(Cz0[1])+Nj[2]*len(Cz0[2])+Nj[3]*len(Cz0[3])] = 0
Gj[Nj[0]*len(Cz0[0])+Nj[1]*len(Cz0[1])+Nj[2]*len(Cz0[2])+Nj[3]*len(Cz0[3]):Nj[0]*len(Cz0[0])+Nj[1]*len(Cz0[1])+Nj[2]*len(Cz0[2])+Nj[3]*len(Cz0[3])+Nj[4]*len(Cz0[4])] = 0
# a geometry problem without any traction
Hj = Gj
end_prepare = time.perf_counter()
time_prepare = end_prepare-start_prepare
print(f"  Solution preparation, time = {time_prepare:.4f} seconds")
# global accumulative method
print("Global accumulative method")
start_global = time.perf_counter()
AA2t = np.zeros((zlength*2,4*Klength), dtype = np.float64)
startj1 = 0
startj = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		endj1 = startj1+lengthj
		startj2 = endj1
		endj2 = startj2+lengthj	
		startJ = 0
		startJ1 = 0
		for J in range(5):
				lengthJ = len(X[J])
				endJ = startJ+lengthJ
				endJ1 = startJ1+lengthJ
				startJ2 = endJ1
				endJ2 = startJ2+lengthJ
				startJ3 = endJ2
				endJ3 = startJ3+lengthJ
				startJ4 = endJ3
				endJ4 = startJ4+lengthJ
				AA2t[startj1:endj1,startJ1:endJ1] = np.real(f1jJk[startj:endj,startJ:endJ])
				AA2t[startj1:endj1,startJ2:endJ2] = np.imag(f1jJk[startj:endj,startJ:endJ])
				AA2t[startj1:endj1,startJ3:endJ3] = np.real(f2jJk[startj:endj,startJ:endJ])
				AA2t[startj1:endj1,startJ4:endJ4] = np.imag(f2jJk[startj:endj,startJ:endJ])
				AA2t[startj2:endj2,startJ1:endJ1] = np.imag(f3jJk[startj:endj,startJ:endJ])
				AA2t[startj2:endj2,startJ2:endJ2] = np.real(f3jJk[startj:endj,startJ:endJ])
				AA2t[startj2:endj2,startJ3:endJ3] = np.imag(f4jJk[startj:endj,startJ:endJ])
				AA2t[startj2:endj2,startJ4:endJ4] = np.real(f4jJk[startj:endj,startJ:endJ])
				startJ1 = endJ4
				startJ += lengthJ
		startj1 = endj2
		startj += lengthj
del f1jJk, f2jJk, f3jJk, f4jJk
BB2t = np.zeros((2*zlength), dtype = np.float64)
X2t = np.ones((4*Klength), dtype = np.float64)
BB2t = AA2t @ X2t
PP = np.zeros((zlength*2,2), dtype = np.float64)
startj = 0
startj1 = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		endj1 = startj1+lengthj
		startj2 = endj1
		endj2 = startj2+lengthj
		PP[startj1:endj1,0] = (1-lambdajr[j])*Lr*KG**(-1)*np.real(Pjj[startj:endj])
		PP[startj1:endj1,1] = (1-lambdajr[j])*Lr*KG**(-1)*np.imag(Pjj[startj:endj])
		PP[startj2:endj2,0] = -(1-lambdajt[j])*Lt*KG**(-1)*np.imag(Pjj[startj:endj])
		PP[startj2:endj2,1] = (1-lambdajt[j])*Lt*KG**(-1)*np.real(Pjj[startj:endj])
		startj1 = endj2
		startj += lengthj
PP_plus = np.zeros((2,zlength*2), dtype = np.float64)
startj1 = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj1 = startj1+lengthj
		startj2 = endj1
		endj2 = startj2+lengthj
		PP_plus[:,startj1:endj2] = np.linalg.pinv(PP[startj1:endj2,:], rcond = 1e-12)
		startj1 = endj2
AA2 = AA2t
del AA2t
startj1 = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj1 = startj1+lengthj
		startj2 = endj1
		endj2 = startj2+lengthj
		startJ1 = 0
		for J in range(5):
				lengthJ = len(X[J])
				endJ1 = startJ1+lengthJ
				startJ2 = endJ1
				endJ2 = startJ2+lengthJ
				startJ3 = endJ2
				endJ3 = startJ3+lengthJ
				startJ4 = endJ3
				endJ4 = startJ4+lengthJ
				AA2[startj1:endj2,startJ1:endJ4] -= PP[startj1:endj2,:] @ (PP_plus[:,startj1:endj2] @ AA2[startj1:endj2,startJ1:endJ4])
				startJ1 = endJ4
		startj1 = endj2
BB2 = np.zeros((2*zlength), dtype = np.float64)
BB2 = BB2t
del BB2t
startj1 = 0
for j in range(5):
		lengthj = Nj[j]*len(Cz0[j])
		endj1 = startj1+lengthj
		startj2 = endj1
		endj2 = startj2+lengthj
		BB2[startj1:endj2] -= PP[startj1:endj2,:] @ (PP_plus[:,startj1:endj2] @ BB2[startj1:endj2])
		startj1 = endj2
print(f"  Shape of A2 = {AA2.shape}")
kappa2A2 = np.linalg.cond(AA2)
print(f"  Condition number of A2 = {kappa2A2:.4e}")
AA2p = np.linalg.pinv(AA2, rcond = 1e-12)
X2 = np.zeros((4*Klength), dtype = np.float64)
X2 = AA2p @ BB2
X2j0 = np.zeros((4*len(X[0])), dtype = np.float64)
X2j1 = np.zeros((4*len(X[1])), dtype = np.float64)
X2j2 = np.zeros((4*len(X[2])), dtype = np.float64)
X2j3 = np.zeros((4*len(X[3])), dtype = np.float64)
X2j4 = np.zeros((4*len(X[4])), dtype = np.float64)
X2j = [X2j0, X2j1, X2j2, X2j3, X2j4]
startj = 0
for j in range(5):
        lengthj = 4*len(X[j])
        endj = startj+lengthj
        X2j[j] = X2[startj:endj]
        startj += lengthj
BB2_after = AA2 @ X2
BB2_residual = BB2_after - BB2
res_global0 = np.zeros((2*Nj[0]*len(Cz0[0])), dtype = np.float64)
res_global1 = np.zeros((2*Nj[1]*len(Cz0[1])), dtype = np.float64)
res_global2 = np.zeros((2*Nj[2]*len(Cz0[2])), dtype = np.float64)
res_global3 = np.zeros((2*Nj[3]*len(Cz0[3])), dtype = np.float64)
res_global4 = np.zeros((2*Nj[4]*len(Cz0[4])), dtype = np.float64)
res_globalj = [res_global0, res_global1, res_global2, res_global3, res_global4]
startj = 0
for j in range(5):
		lengthj = 2*Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		res_globalj[j] = BB2_residual[startj:endj]
		startj += lengthj
res_mean_global = np.zeros((5), dtype = np.float64)
res_max_global = np.zeros((5), dtype = np.float64)
res_total_global = np.zeros((5), dtype = np.float64)
for j in range(5):
		res_mean_global[j] = np.sqrt(np.mean((res_globalj[j])**2))
		res_max_global[j] = np.max(np.abs(res_globalj[j]))
		res_total_global[j] = np.sum(np.abs(res_globalj[j])**2)
		print(f"  Global, Periphery {j}, max difference = {res_max_global[j]:.4e}")
end_global = time.perf_counter()
time_global = end_global-start_global
print(f"  Global accumulative method, time = {time_global:.4f} seconds")
# Schwarz alternating method
print("Schwarz alternating method")
start_schwarz = time.perf_counter()
height = [2*Nj[0]*len(Cz0[0]),2*Nj[1]*len(Cz0[1]),2*Nj[2]*len(Cz0[2]),2*Nj[3]*len(Cz0[3]),2*Nj[4]*len(Cz0[4])]
width = [4*(len(X[0])),4*(len(X[1])),4*(len(X[2])),4*(len(X[3])),4*(len(X[4]))]
row_bounds = np.insert(np.cumsum(height), 0, 0)
column_bounds = np.insert(np.cumsum(width), 0, 0)
AA2_blocks = [[AA2[row_bounds[i]:row_bounds[i+1], column_bounds[j]:column_bounds[j+1]] for j in range(5)] for i in range(5)]
AA2_restore = np.block(AA2_blocks)
if np.allclose(AA2, AA2_restore):
		print(f"  {'='*20}\n  Correct segments of A2")
else:
		print("  {'='*20}\n  Wrong segments of A2")
		diff = np.abs(AA2_restore - AA2)
		maxerror_idx = np.unravel_index(np.argmax(diff), diff.shape)
		print(f"  Maximum error exists: {maxerror_idx}, error = {diff[maxerror_idx]}")
del AA2_restore
kappa2AA2j = np.zeros((5), dtype = np.float64)
AA200 = np.zeros((2*Nj[0]*len(Cz0[0]),4*len(X[0])), dtype = np.float64)
AA211 = np.zeros((2*Nj[1]*len(Cz0[1]),4*len(X[1])), dtype = np.float64)
AA222 = np.zeros((2*Nj[2]*len(Cz0[2]),4*len(X[2])), dtype = np.float64)
AA233 = np.zeros((2*Nj[3]*len(Cz0[3]),4*len(X[3])), dtype = np.float64)
AA244 = np.zeros((2*Nj[4]*len(Cz0[4]),4*len(X[4])), dtype = np.float64)
AA2jj = [AA200, AA211, AA222, AA233, AA244]
for j in range(5):
		kappa2AA2j[j] = np.linalg.cond(AA2_blocks[j][j])
		print(f"  Condition number of {j+1}th step = {kappa2AA2j[j]:.4e}")
		AA2jj[j] = np.linalg.pinv(AA2_blocks[j][j], rcond = 1e-12)
BB20 = np.zeros((2*Nj[0]*len(Cz0[0])), dtype = np.float64)
BB21 = np.zeros((2*Nj[1]*len(Cz0[1])), dtype = np.float64)
BB22 = np.zeros((2*Nj[2]*len(Cz0[2])), dtype = np.float64)
BB23 = np.zeros((2*Nj[3]*len(Cz0[3])), dtype = np.float64)
BB24 = np.zeros((2*Nj[4]*len(Cz0[4])), dtype = np.float64)
BB20 = BB2[:2*Nj[0]*len(Cz0[0])]
BB21 = BB2[2*Nj[0]*len(Cz0[0]):2*Nj[0]*len(Cz0[0])+2*Nj[1]*len(Cz0[1])]
BB22 = BB2[2*Nj[0]*len(Cz0[0])+2*Nj[1]*len(Cz0[1]):2*Nj[0]*len(Cz0[0])+2*Nj[1]*len(Cz0[1])+2*Nj[2]*len(Cz0[2])]
BB23 = BB2[2*Nj[0]*len(Cz0[0])+2*Nj[1]*len(Cz0[1])+2*Nj[2]*len(Cz0[2]):2*Nj[0]*len(Cz0[0])+2*Nj[1]*len(Cz0[1])+2*Nj[2]*len(Cz0[2])+2*Nj[3]*len(Cz0[3])]
BB24 = BB2[2*Nj[0]*len(Cz0[0])+2*Nj[1]*len(Cz0[1])+2*Nj[2]*len(Cz0[2])+2*Nj[3]*len(Cz0[3]):2*Nj[0]*len(Cz0[0])+2*Nj[1]*len(Cz0[1])+2*Nj[2]*len(Cz0[2])+2*Nj[3]*len(Cz0[3])+2*Nj[4]*len(Cz0[4])]
BB2j = [BB20, BB21, BB22, BB23, BB24]
tol_itr = 1e-12
max_itr = 1000
X20 = np.zeros((4*(len(X[0]))), dtype = np.float64)
X21 = np.zeros((4*(len(X[1]))), dtype = np.float64)
X22 = np.zeros((4*(len(X[2]))), dtype = np.float64)
X23 = np.zeros((4*(len(X[3]))), dtype = np.float64)
X24 = np.zeros((4*(len(X[4]))), dtype = np.float64)
X2J = [X20, X21, X22, X23, X24]
X210 = np.zeros((4*(len(X[0]))), dtype = np.float64)
X211 = np.zeros((4*(len(X[1]))), dtype = np.float64)
X212 = np.zeros((4*(len(X[2]))), dtype = np.float64)
X213 = np.zeros((4*(len(X[3]))), dtype = np.float64)
X214 = np.zeros((4*(len(X[4]))), dtype = np.float64)
X21J = [X210, X211, X212, X213, X214]
i = 0
itr_rep = np.zeros((max_itr), dtype = int)
err_rep = np.zeros((max_itr), dtype = np.float64)
for i in range(max_itr):
        max_err = 0.
        itr_rep[i] = i+1
        for j in range(5):
                AX = np.zeros((2*Nj[j]*len(Cz0[j])), dtype = np.float64)
                for J in range(5):
                        if J != j:
                            AX += AA2_blocks[j][J] @ X21J[J]
                if i == 0:
                        BX = BB2j[j] - AX
                else:
                        BX = -AX
                X21J[j] = AA2jj[j] @ BX
                err1 = np.max(np.abs(X21J[j]))
                max_err = np.maximum(max_err, err1)
                X2J[j] += X21J[j]
                #print(f"      Iteration {i+1}, step {j+1}, error = {err1:.4e}")
        err_rep[i] = max_err
        #print(f"    Iteration {i+1}, error = {err_rep[i]:.4e}")
        if max_err < tol_itr:
                print(f"  Iteration stops at iteration {i+1}, max error = {err_rep[i]:.4e}")
                break
else:
        print("  No convergence for maximum iteration reps")
del AA2_blocks
mask = np.abs(err_rep) > 1e-13
err_rep_no_zero = err_rep[mask]
itr_rep_no_zero = itr_rep[mask]
X2S = np.zeros((4*Klength), dtype = np.float64)
X2S[:4*(len(X[0]))] = X2J[0]
X2S[4*(len(X[0])):4*(len(X[0]))+4*(len(X[1]))] = X2J[1]
X2S[4*(len(X[0]))+4*(len(X[1])):4*(len(X[0]))+4*(len(X[1]))+4*(len(X[2]))] = X2J[2]
X2S[4*(len(X[0]))+4*(len(X[1]))+4*(len(X[2])):4*(len(X[0]))+4*(len(X[1]))+4*(len(X[2]))+4*(len(X[3]))] = X2J[3]
X2S[4*(len(X[0]))+4*(len(X[1]))+4*(len(X[2]))+4*(len(X[3])):4*(len(X[0]))+4*(len(X[1]))+4*(len(X[2]))+4*(len(X[3]))+4*(len(X[4]))] = X2J[4]
BB2j_after = AA2 @ X2S
BB2j_residual = BB2j_after - BB2
res_Schwarz0 = np.zeros((2*Nj[0]*len(Cz0[0])), dtype = np.float64)
res_Schwarz1 = np.zeros((2*Nj[1]*len(Cz0[1])), dtype = np.float64)
res_Schwarz2 = np.zeros((2*Nj[2]*len(Cz0[2])), dtype = np.float64)
res_Schwarz3 = np.zeros((2*Nj[3]*len(Cz0[3])), dtype = np.float64)
res_Schwarz4 = np.zeros((2*Nj[4]*len(Cz0[4])), dtype = np.float64)
res_Schwarzj = [res_Schwarz0, res_Schwarz1, res_Schwarz2, res_Schwarz3, res_Schwarz4]
startj = 0
for j in range(5):
		lengthj = 2*Nj[j]*len(Cz0[j])
		endj = startj+lengthj
		res_Schwarzj[j] = BB2j_residual[startj:endj]
		startj += lengthj
res_mean_schwarz = np.zeros((5), dtype = np.float64)
res_max_schwarz = np.zeros((5), dtype = np.float64)
res_total_schwarz = np.zeros((5), dtype = np.float64)
for j in range(5):
		res_mean_schwarz[j] = np.sqrt(np.mean((res_Schwarzj[j])**2))
		res_max_schwarz[j] = np.max(np.abs(res_Schwarzj[j]))
		res_total_schwarz[j] = np.sum(np.abs(res_Schwarzj[j])**2)
		print(f"  Schwarz, Periphery {j}, max difference = {res_max_schwarz[j]:.4e}")
end_schwarz = time.perf_counter()
time_schwarz = end_schwarz-start_schwarz
print(f"  Schwarz alternating method, time = {time_schwarz:.4f} seconds")
# comparison
X2max = np.max(np.abs(X2 - X2S))
print(f"  Maximum solution difference between global accumulative method and Schwarz alternating method = {X2max:.4e}")
X2d0g = np.zeros((4*len(X[0])), dtype = np.float64)
X2d1g = np.zeros((4*len(X[1])), dtype = np.float64)
X2d2g = np.zeros((4*len(X[2])), dtype = np.float64)
X2d3g = np.zeros((4*len(X[3])), dtype = np.float64)
X2d4g = np.zeros((4*len(X[4])), dtype = np.float64)
X2dg = [X2d0g, X2d1g, X2d2g, X2d3g, X2d4g]
X2d0S = np.zeros((4*len(X[0])), dtype = np.float64)
X2d1S = np.zeros((4*len(X[1])), dtype = np.float64)
X2d2S = np.zeros((4*len(X[2])), dtype = np.float64)
X2d3S = np.zeros((4*len(X[3])), dtype = np.float64)
X2d4S = np.zeros((4*len(X[4])), dtype = np.float64)
X2dS = [X2d0S, X2d1S, X2d2S, X2d3S, X2d4S]
X20t = np.ones((4*len(X[0])), dtype = np.float64)
X21t = np.ones((4*len(X[1])), dtype = np.float64)
X22t = np.ones((4*len(X[2])), dtype = np.float64)
X23t = np.ones((4*len(X[3])), dtype = np.float64)
X24t = np.ones((4*len(X[4])), dtype = np.float64)
X2t = [X20t, X21t, X22t, X23t, X24t]
X2maxgj = np.zeros((5), dtype = np.float64)
X2maxSj = np.zeros((5), dtype = np.float64)
for j in range(5):
        X2dS[j] = X2J[j]-X2t[j]
        X2maxSj[j] = np.max(np.abs(X2dS[j]))
        print(f"    Maximum solution difference of Schwarz alternating method for {j}th periphery = {X2maxSj[j]:.4e}")
for j in range(5):
        X2dg[j] = X2j[j]-X2t[j]
        X2maxgj[j] = np.max(np.abs(X2dg[j]))
        print(f"    Maximum solution difference of global accumulative method for {j}th periphery = {X2maxgj[j]:.4e}")
