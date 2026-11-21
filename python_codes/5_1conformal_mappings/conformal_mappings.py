import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# Geometry
# -----------------------------------------------------------------
from functions_conformal_mapping import arc_points, charge_point, forward_mapping_solution, backward_mapping_solution, forward_mapping, backward_mapping, inside_C1, inside_C2, inside_C3, inside_C4, inside_C0
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
# arc radius 0.1
C0r = 0.5
# Straight segment length 20-2*0.02
x0C0local, x1C0local = -10, 10
y0C0local, y1C0local = -10, 10
# Straight segment sample point quantity
C0local_horizontal_num = int(40*(10-C0r))
C0local_vertical_num = int(40*(10-C0r))
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
C0local_periphery_z_shift = np.roll(C0local_periphery_z, -1)
dz0 = C0local_periphery_z_shift - C0local_periphery_z
length_dz0 = np.abs(dz0)
# Charge point distribution along C0 periphery, also ant-clockwise
C0local_periphery_X, C0local_periphery_charge_point = charge_point(C0local_periphery_z, RjC0)
# Forward mapping solution for C0
QC0, Gamma_0, kappa2C0 = forward_mapping_solution(C0local_periphery_z, C0local_periphery_X, 0, N, z_0c)
# Forward conformal mapping grid, in rectangular form
xC0 = np.linspace(-10, 10, 200)
yC0 = np.linspace(-10, 10, 200)
xgridC0, ygridC0 = np.meshgrid(xC0, yC0)
zgridC0 = xgridC0 + 1j * ygridC0
# Determine whether point in z plane is inside of C0, using function inside_C0
point_inside_C0 = inside_C0(zgridC0, C0r, x0C0local, x1C0local, y0C0local, y1C0local)
z_inside_C0 = zgridC0[point_inside_C0]
x_inside_C0 = np.real(z_inside_C0)
y_inside_C0 = np.imag(z_inside_C0)
# Compute forward mapping points inside of C0
forward_mapping_inside_C0 = forward_mapping(z_inside_C0, C0local_periphery_X, Gamma_0, QC0, 0, z_0c, N)
forward_mapping_inside_C0x = np.real(forward_mapping_inside_C0)
forward_mapping_inside_C0y = np.imag(forward_mapping_inside_C0)
# Compute forward mapping point along C0
forward_mapping_along_C0 = forward_mapping(C0local_periphery_z, C0local_periphery_X, Gamma_0, QC0, 0, z_0c, N)
forward_mapping_along_C0x = np.real(forward_mapping_along_C0)
forward_mapping_along_C0y = np.imag(forward_mapping_along_C0)
# Forward conformal mapping for C0 finishes
# Backward conformal mapping for C0 starts
c0local_periphery_zeta = forward_mapping_along_C0
c0local_periphery_zeta_shift = np.roll(c0local_periphery_zeta, -1)
dzc0 = c0local_periphery_zeta_shift - c0local_periphery_zeta
length_dzc0 = np.abs(dzc0) * 10.
# Compute charge points along mapped unit-radius periphery in mapping plane
c0local_periphery_x, c0local_periphery_charge_point = charge_point(c0local_periphery_zeta, rjc0)
# Backward mapping solution for c0
qc0, kappa2c0 = backward_mapping_solution(C0local_periphery_z, c0local_periphery_zeta, c0local_periphery_x, 0, N)
# Backward conformal mapping grid, in polar form
r0_dist = np.linspace(0., 1., 51)
theta0_dist = np.linspace(0., 2. * np.pi, 361)
theta0_dist = np.r_[theta0_dist, theta0_dist[0]]
r0_grid, theta0_grid = np.meshgrid(r0_dist, theta0_dist)
zeta0_grid = r0_grid * np.exp(1j * theta0_grid)
# Backward conformal mapping in original plane
z0_grid = backward_mapping(zeta0_grid, c0local_periphery_x, qc0, 0, N)
z0_grid = z0_grid[:-1, :]
# Geometry.C1
# Forward assigment factor
RjC1 = 2.0
# Backward assigment factor
rjc1 = 2.0
# Reference point should be located at center of the local coordinate system due to symmetry about both x and y axes
z_1c = 0. + 1j * 0.
# Geometry of C1 is an ellipse with radius 3
theta_num = 360
a1 = 3
b1 = 3
theta = np.linspace(2.*np.pi, 0, theta_num, endpoint = False)
C1local_periphery_x = a1 * np.cos(theta)
C1local_periphery_y = b1 * np.sin(theta)
C1local_periphery = np.column_stack([C1local_periphery_x,C1local_periphery_y])
C1local_periphery_z = C1local_periphery_x + 1j * C1local_periphery_y
# Forward conformal mapping
C1local_periphery_X, C1local_periphery_charge_point = charge_point(C1local_periphery_z, RjC1)
# Forward mapping solution for C1
QC1, Gamma_1, kappa2C1 = forward_mapping_solution(C1local_periphery_z, C1local_periphery_X, 1, N, z_1c)
# Forward conformal mapping grid, in rectangular form
x0C1local = - a1
x1C1local = a1
y0C1local = - a1
y1C1local = a1
xystep1 = a1/10
xC1 = np.arange(x0C1local*3.0, x1C1local*3.0 + xystep1, xystep1)
yC1 = np.arange(y0C1local*3.0, y1C1local*3.0 + xystep1, xystep1)
xgridC1, ygridC1 = np.meshgrid(xC1, yC1)
zgridC1 = xgridC1 + 1j * ygridC1
# Determine whether point in z plane is inside of C1, using function inside_C1
point_inside_C1 = inside_C1(zgridC1, a1)
point_outside_C1 = ~point_inside_C1
z_outside_C1 = zgridC1[point_outside_C1]
x_outside_C1 = np.real(z_outside_C1)
y_outside_C1 = np.imag(z_outside_C1)
# Compute forward mapping points outside of C1
forward_mapping_outside_C1 = forward_mapping(z_outside_C1, C1local_periphery_X, Gamma_1, QC1, 1, z_1c, N)
forward_mapping_outside_C1x = np.real(forward_mapping_outside_C1)
forward_mapping_outside_C1y = np.imag(forward_mapping_outside_C1)
# Compute forward mapping point along C1
forward_mapping_along_C1 = forward_mapping(C1local_periphery_z, C1local_periphery_X, Gamma_1, QC1, 1, z_1c, N)
forward_mapping_along_C1x = np.real(forward_mapping_along_C1)
forward_mapping_along_C1y = np.imag(forward_mapping_along_C1)
# Forward conformal mapping for C1 finishes
# Backward conformal mapping for C1 starts
c1local_periphery_zeta = forward_mapping_along_C1
# Compute charge points along mapped unit-radius periphery in mapping plane
c1local_periphery_x, c1local_periphery_charge_point = charge_point(c1local_periphery_zeta, rjc1)
# Backward mapping solution for c1
qc1, kappa2c1 = backward_mapping_solution(C1local_periphery_z, c1local_periphery_zeta, c1local_periphery_x, 1, N)
# Backward conformal mapping grid, in polar form
r1_dist = np.linspace(1., 6., 51)
theta1_dist = np.linspace(0., 2. * np.pi, 121)
theta1_dist = np.r_[theta1_dist, theta1_dist[0]]
r1_grid, theta1_grid = np.meshgrid(r1_dist, theta1_dist)
zeta1_grid = r1_grid * np.exp(1j * theta1_grid)
# Backward conformal mapping in original plane
z1_grid = backward_mapping(zeta1_grid, c1local_periphery_x, qc1, 1, N)
z1_grid = z1_grid[:-1, :]
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
C2local_periphery_X, C2local_periphery_charge_point = charge_point(C2local_periphery_z, RjC2)
# Forward mapping solution for C2
QC2, Gamma_2, kappa2C2 = forward_mapping_solution(C2local_periphery_z, C2local_periphery_X, 2, N, z_2c)
# Forward conformal mapping grid, in rectangular form
xystep2 = C2r
xC2 = np.arange(x0C2local*3.0, x1C2local*3.0 + xystep2, xystep2)
yC2 = np.arange(y0C2local*3.0, y1C2local*3.0 + xystep2, xystep2)
xgridC2, ygridC2 = np.meshgrid(xC2, yC2)
zgridC2 = xgridC2 + 1j * ygridC2
# Determine whether point in z plane is inside of C2, using function inside_C2
point_inside_C2 = inside_C2(zgridC2, C2r, x0C2local, x1C2local, y0C2local, y1C2local)
point_outside_C2 = ~point_inside_C2
z_outside_C2 = zgridC2[point_outside_C2]
x_outside_C2 = np.real(z_outside_C2)
y_outside_C2 = np.imag(z_outside_C2)
# Compute forward mapping points outside of C2
forward_mapping_outside_C2 = forward_mapping(z_outside_C2, C2local_periphery_X, Gamma_2, QC2, 2, z_2c, N)
forward_mapping_outside_C2x = np.real(forward_mapping_outside_C2)
forward_mapping_outside_C2y = np.imag(forward_mapping_outside_C2)
# Compute forward mapping point along C2
forward_mapping_along_C2 = forward_mapping(C2local_periphery_z, C2local_periphery_X, Gamma_2, QC2, 2, z_2c, N)
forward_mapping_along_C2x = np.real(forward_mapping_along_C2)
forward_mapping_along_C2y = np.imag(forward_mapping_along_C2)
# Forward conformal mapping for C2 finishes
# Backward conformal mapping for C2 starts
c2local_periphery_zeta = forward_mapping_along_C2
# Compute charge points along mapped unit-radius periphery in mapping plane
c2local_periphery_x, c2local_periphery_charge_point = charge_point(c2local_periphery_zeta, rjc2)
# Backward mapping solution for c2
qc2, kappa2c2 = backward_mapping_solution(C2local_periphery_z, c2local_periphery_zeta, c2local_periphery_x, 2, N)
# Backward conformal mapping grid, in polar form
r2_dist = np.linspace(1., 6., 51)
theta2_dist = np.linspace(0., 2. * np.pi, 121)
theta2_dist = np.r_[theta2_dist, theta2_dist[0]]
r2_grid, theta2_grid = np.meshgrid(r2_dist, theta2_dist)
zeta2_grid = r2_grid * np.exp(1j * theta2_grid)
# Backward conformal mapping in original plane
z2_grid = backward_mapping(zeta2_grid, c2local_periphery_x, qc2, 2, N)
z2_grid = z2_grid[:-1, :]
# Geometry.C3
# Forward assigment factor
RjC3 = 2.0
# Backward assigment factor
rjc3 = 2.0
# Reference point should be located at center of the local coordinate system due to symmetry about both x and y axes
z_3c = 0. + 1j * 0.
# Geometry of C3 is an ellipse with x radius 3 and y radius 2
theta_num = 360
a3 = 3.
b3 = 2
theta = np.linspace(2.*np.pi, 0, theta_num, endpoint = False)
C3local_periphery_x = a3 * np.cos(theta)
C3local_periphery_y = b3 * np.sin(theta)
C3local_periphery = np.column_stack([C3local_periphery_x,C3local_periphery_y])
C3local_periphery_z = C3local_periphery_x + 1j * C3local_periphery_y
# Forward conformal mapping
C3local_periphery_X, C3local_periphery_charge_point = charge_point(C3local_periphery_z, RjC3)
# Forward mapping solution for C3
QC3, Gamma_3, kappa2C3 = forward_mapping_solution(C3local_periphery_z, C3local_periphery_X, 3, N, z_3c)
# Forward conformal mapping grid, in rectangular form
x0C3local = - a3
x1C3local = a3
y0C3local = - b3
y1C3local = b3
xystep3 = (a3+b3)/20
xC3 = np.arange(x0C3local*3.0, x1C3local*3.0 + xystep3, xystep3)
yC3 = np.arange(y0C3local*3.0, y1C3local*3.0 + xystep3, xystep3)
xgridC3, ygridC3 = np.meshgrid(xC3, yC3)
zgridC3 = xgridC3 + 1j * ygridC3
# Determine whether point in z/w plane is inside of C3, using function inside_C3
point_inside_C3 = inside_C3(zgridC3, a3, b3)
point_outside_C3 = ~point_inside_C3
z_outside_C3 = zgridC3[point_outside_C3]
x_outside_C3 = np.real(z_outside_C3)
y_outside_C3 = np.imag(z_outside_C3)
# Compute forward mapping points outside of C3
forward_mapping_outside_C3 = forward_mapping(z_outside_C3, C3local_periphery_X, Gamma_3, QC3, 3, z_3c, N)
forward_mapping_outside_C3x = np.real(forward_mapping_outside_C3)
forward_mapping_outside_C3y = np.imag(forward_mapping_outside_C3)
# Compute forward mapping point along C3
forward_mapping_along_C3 = forward_mapping(C3local_periphery_z, C3local_periphery_X, Gamma_3, QC3, 3, z_3c, N)
forward_mapping_along_C3x = np.real(forward_mapping_along_C3)
forward_mapping_along_C3y = np.imag(forward_mapping_along_C3)
# Forward conformal mapping for C3 finishes
# Backward conformal mapping for C3 starts
c3local_periphery_zeta = forward_mapping_along_C3
# Compute charge points along mapped unit-radius periphery in mapping plane
c3local_periphery_x, c3local_periphery_charge_point = charge_point(c3local_periphery_zeta, rjc3)
# Backward mapping solution for c3
qc3, kappa2c3 = backward_mapping_solution(C3local_periphery_z, c3local_periphery_zeta, c3local_periphery_x, 3, N)
# Backward conformal mapping grid, in polar form
r3_dist = np.linspace(1., 6., 51)
theta3_dist = np.linspace(0., 2. * np.pi, 121)
theta3_dist = np.r_[theta3_dist, theta3_dist[0]]
r3_grid, theta3_grid = np.meshgrid(r3_dist, theta3_dist)
zeta3_grid = r3_grid * np.exp(1j * theta3_grid)
# Backward conformal mapping in original plane
z3_grid = backward_mapping(zeta3_grid, c3local_periphery_x, qc3, 3, N)
z3_grid = z3_grid[:-1, :]
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
theta_num = 360
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
C4local_periphery_X, C4local_periphery_charge_point = charge_point(C4local_periphery_z, RjC4)
# Forward mapping solution for C4
QC4, Gamma_4, kappa2C4 = forward_mapping_solution(C4local_periphery_z, C4local_periphery_X, 4, N, z_4c)
# Forward conformal mapping grid, in rectangular form
x0C4local = - a4
x1C4local = a4
y0C4local = - b4lw
y1C4local = b4up
xystep4 = (a4+b4lw)/20
xC4 = np.arange(x0C4local*3.0, x1C4local*3.0 + xystep4, xystep4)
yC4 = np.arange(y0C4local*3.0, y1C4local*3.0 + xystep4, xystep4)
xgridC4, ygridC4 = np.meshgrid(xC4, yC4)
zgridC4 = xgridC4 + 1j * ygridC4
# Determine whether point in z/w plane is inside of C4, using function inside_C4
point_inside_C4 = inside_C4(zgridC4, a4, b4up, b4lw)
point_outside_C4 = ~point_inside_C4
z_outside_C4 = zgridC4[point_outside_C4]
x_outside_C4 = np.real(z_outside_C4)
y_outside_C4 = np.imag(z_outside_C4)
# Compute forward mapping points outside of C4
forward_mapping_outside_C4 = forward_mapping(z_outside_C4, C4local_periphery_X, Gamma_4, QC4, 4, z_4c, N)
forward_mapping_outside_C4x = np.real(forward_mapping_outside_C4)
forward_mapping_outside_C4y = np.imag(forward_mapping_outside_C4)
# Compute forward mapping point along C4
forward_mapping_along_C4 = forward_mapping(C4local_periphery_z, C4local_periphery_X, Gamma_4, QC4, 4, z_4c, N)
forward_mapping_along_C4x = np.real(forward_mapping_along_C4)
forward_mapping_along_C4y = np.imag(forward_mapping_along_C4)
# Forward conformal mapping for C4 finishes
# Backward conformal mapping for C4 starts
c4local_periphery_zeta = forward_mapping_along_C4
# Compute charge points along mapped unit-radius periphery in mapping plane
c4local_periphery_x, c4local_periphery_charge_point = charge_point(c4local_periphery_zeta, rjc4)
# Backward mapping solution for c4
qc4, kappa2c4 = backward_mapping_solution(C4local_periphery_z, c4local_periphery_zeta, c4local_periphery_x, 4, N)
# Backward conformal mapping grid, in polar form
r4_dist = np.linspace(1., 6., 51)
theta4_dist = np.linspace(0., 2. * np.pi, 121)
theta4_dist = np.r_[theta4_dist, theta4_dist[0]]
r4_grid, theta4_grid = np.meshgrid(r4_dist, theta4_dist)
zeta4_grid = r4_grid * np.exp(1j * theta4_grid)
# Backward conformal mapping in original plane
z4_grid = backward_mapping(zeta4_grid, c4local_periphery_x, qc4, 4, N)
z4_grid = z4_grid[:-1, :]
# ----------------------------------------------------------
z_C1 = C1local_periphery_z
z_C2 = C2local_periphery_z
z_C3 = C3local_periphery_z
z_C4 = C4local_periphery_z
zO1 = -5 + 5j
zO2 = 5 + 5j
zO3 = -5 - 5j
zO4 = 5 - 5j
# ----------------------------------------------------------
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

fig = plt.figure(figsize=(40, 20))

outer_gs = gridspec.GridSpec(
    nrows=1, ncols=2, figure=fig,
    width_ratios=[1, 1], wspace=0.25,
)
right_gs = outer_gs[0, 1].subgridspec(
    nrows=2, ncols=2, hspace=0.35, wspace=0.25,
)

ax0 = fig.add_subplot(outer_gs[:, 0])
num_theta0, num_r0 = z0_grid.shape
step_r0 = 1
step_theta0 = 1

for i in range(0, num_theta0, step_theta0):
    ax0.plot(np.real(z0_grid[i, :]), np.imag(z0_grid[i, :]), "c-", lw=1.5, alpha=0.6)
for j in range(0, num_r0, step_r0):
    ax0.plot(np.real(z0_grid[:, j]), np.imag(z0_grid[:, j]), "c-", lw=1.5, alpha=0.6)

ax0.plot(np.real(C0local_periphery_z), np.imag(C0local_periphery_z), "b-", lw=1.5)
ax0.plot(np.real(z_C1 + zO1), np.imag(z_C1 + zO1), "--", color="blue", lw=1.5)
ax0.plot(np.real(z_C2 + zO2), np.imag(z_C2 + zO2), "--", color="blue", lw=1.5)
ax0.plot(np.real(z_C3 + zO3), np.imag(z_C3 + zO3), "--", color="blue", lw=1.5)
ax0.plot(np.real(z_C4 + zO4), np.imag(z_C4 + zO4), "--", color="blue", lw=1.5)

ax0.set_aspect("equal")
ax0.grid(True, alpha=0.3)
ax0.set_xlabel(r"$x_{0}(x)$", fontsize=30)
ax0.set_ylabel(r"$y_{0}(y)$", fontsize=30)
ax0.set_xlim(-10.1, 10.1)
ax0.set_ylim(-10.1, 10.1)
ax0.set_xticks(np.arange(-10, 10.5, 2))
ax0.set_yticks(np.arange(-10, 10.5, 2))
ax0.tick_params(axis='both', labelsize=30)
ax0.set_title(r"(a) Subdomain $\overline{\Omega}_{0}$ and interior periphery locations", fontsize=30)


plot_configs = [
    # (zeta_grid, z_grid, step_theta, step_r, special_z, special_color, subtitle)
    (z1_grid, z1_grid, 1, 1, C1local_periphery_z, "blue", r"(b) Subdomain ${\overline{\Omega}}_{1}$"),
    (z2_grid, z2_grid, 1, 1, C2local_periphery_z, "blue", r"(c) Subdomain ${\overline{\Omega}}_{2}$"),
    (z3_grid, z3_grid, 1, 1, C3local_periphery_z, "blue", r"(d) Subdomain ${\overline{\Omega}}_{3}$"),
    (z4_grid, z4_grid, 1, 1, C4local_periphery_z, "blue", r"(e) Subdomain ${\overline{\Omega}}_{4}$"),
]

for idx1 in range(2):
    for idx2 in range(2):
        ax = fig.add_subplot(right_gs[idx1, idx2])
        zeta_grid, z_data, step_theta, step_r, special_z, color_sp, sub_title = plot_configs[idx1*2+idx2]
        num_theta, num_r = zeta_grid.shape

        # grid in theta direction
        for i in range(0, num_theta, step_theta):
            ax.plot(np.real(z_data[i, :]), np.imag(z_data[i, :]),
                    "-", color="c", lw=1.5, alpha=0.6)
        # grid in r direction
        for j in range(0, num_r, step_r):
            ax.plot(np.real(z_data[:, j]), np.imag(z_data[:, j]),
                    "-", color="c", lw=1.5, alpha=0.6)

        ax.plot(np.real(z_data[0, :]), np.imag(z_data[0, :]),
                "-", color="c", lw=1.5, alpha=0.6)
        ax.plot(np.real(z_data[:, 0]), np.imag(z_data[:, 0]),
                "-", color="c", lw=1.5, alpha=0.6)
        ax.set_xlabel(rf"$x_{{{idx1*2+idx2+1}}}$", fontsize=30)
        ax.set_ylabel(rf"$y_{{{idx1*2+idx2+1}}}$", fontsize=30)

        ax.plot(np.real(special_z), np.imag(special_z), "-", color=color_sp, lw=1.5)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xticks(np.arange(-5, 6, 1))
        ax.set_yticks(np.arange(-5, 6, 1))
        ax.tick_params(axis='both', labelsize=24)
        ax.set_aspect("equal")
        ax.set_title(sub_title, fontsize=30)

plt.savefig(
    "/home/lin/frontend/papers/GAM/fig_mapping.pdf",
    bbox_inches="tight"
)
plt.close(fig)

print("Error computation")
C0local_periphery_z_half = (C0local_periphery_z + C0local_periphery_z_shift)/2
forward_mapping_along_C0_half = forward_mapping(C0local_periphery_z_half, C0local_periphery_X, Gamma_0, QC0, 0, z_0c, N)
z0_half_back = backward_mapping(forward_mapping_along_C0_half, c0local_periphery_x, qc0, 0, N)
C0_half_max = np.max(np.abs(C0local_periphery_z_half-z0_half_back))/(19.8*np.sqrt(2)+0.2)
print(f"  Maximum relative difference for C0 = {C0_half_max:.4e}")
C1local_periphery_z_shift = np.roll(C1local_periphery_z, -1)
C1local_periphery_z_half = (C1local_periphery_z + C1local_periphery_z_shift)/2
forward_mapping_along_C1_half = forward_mapping(C1local_periphery_z_half, C1local_periphery_X, Gamma_1, QC1, 1, z_1c, N)
z1_half_back = backward_mapping(forward_mapping_along_C1_half, c1local_periphery_x, qc1, 1, N)
C1_half_max = np.max(np.abs(C1local_periphery_z_half-z1_half_back))/3
print(f"  Maximum relative difference for C1 = {C1_half_max:.4e}")
C2local_periphery_z_shift = np.roll(C2local_periphery_z, -1)
C2local_periphery_z_half = (C2local_periphery_z + C2local_periphery_z_shift)/2
forward_mapping_along_C2_half = forward_mapping(C2local_periphery_z_half, C2local_periphery_X, Gamma_2, QC2, 2, z_2c, N)
z2_half_back = backward_mapping(forward_mapping_along_C2_half, c2local_periphery_x, qc2, 2, N)
C2_half_max = np.max(np.abs(C2local_periphery_z_half-z2_half_back))/3
print(f"  Maximum relative difference for C2 = {C2_half_max:.4e}")
C3local_periphery_z_shift = np.roll(C3local_periphery_z, -1)
C3local_periphery_z_half = (C3local_periphery_z + C3local_periphery_z_shift)/2
forward_mapping_along_C3_half = forward_mapping(C3local_periphery_z_half, C3local_periphery_X, Gamma_3, QC3, 3, z_3c, N)
z3_half_back = backward_mapping(forward_mapping_along_C3_half, c3local_periphery_x, qc3, 3, N)
C3_half_max = np.max(np.abs(C3local_periphery_z_half-z3_half_back))/3
print(f"  Maximum relative difference for C3 = {C3_half_max:.4e}")
C4local_periphery_z_shift = np.roll(C4local_periphery_z, -1)
C4local_periphery_z_half = (C4local_periphery_z + C4local_periphery_z_shift)/2
forward_mapping_along_C4_half = forward_mapping(C4local_periphery_z_half, C4local_periphery_X, Gamma_4, QC4, 4, z_4c, N)
z4_half_back = backward_mapping(forward_mapping_along_C4_half, c4local_periphery_x, qc4, 4, N)
C4_half_max = np.max(np.abs(C4local_periphery_z_half-z4_half_back))/3
print(f"  Maximum relative difference for C4 = {C4_half_max:.4e}")

# table
df = pd.DataFrame({
		'$j$': [r'$R_{j}$', r'$r_{j}$', r'$C_{j}^{f}$', r'$C_{j}^{b}$', r'$\varGamma_{j}$', r'$\varepsilon_{j,\frac{1}{2}}$'],
		'$0$': [RjC0, rjc0, kappa2C0, kappa2c0, 'Unapplicable', C0_half_max],
		'$1$': [RjC1, rjc1, kappa2C1, kappa2c1, Gamma_1, C1_half_max],
		'$2$': [RjC2, rjc2, kappa2C2, kappa2c2, Gamma_2, C2_half_max],
		'$3$': [RjC3, rjc3, kappa2C3, kappa2c3, Gamma_3, C3_half_max],
		'$4$': [RjC4, rjc4, kappa2C4, kappa2c4, Gamma_4, C4_half_max],
})
def four_sig(x):
		if isinstance(x, (float, int, np.floating)):
				if x == 0:
						return "0"
				s = f"{x:4e}"
				mantissa, exponent = s.split("e")
				exp_int = int(exponent)
				return f"${mantissa} \\cdot 10^{{{exp_int}}}$"
		return str(x)
df = df.map(four_sig)
latex_code = (
    df.style
    .hide(axis="index")
    .to_latex(
        siunitx=False,
        hrules=True,
        column_format='c|ccccc',
        environment=None,
    )
)

with open("../../tab_mapping.tex", "w", encoding="utf-8") as f:
    f.write(latex_code)
