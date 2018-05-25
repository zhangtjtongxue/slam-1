"""

This test will simulate the SLAM problem:
	* The world will be a quadratic world of size m, containing l landmarks in random positions
	* The robot will be placed in the middle of the world with a heading of 0 (facing east)
	* A series of movements will be generated. The controls specify a speed in which the robot
		moves and a turn angle
	* After moving, measurements containing (range,bearing) will be supplied to the robot.

Author:	C.Rauterberg

"""

import numpy as np
from slam import ExKFSlamKA, EnsKFSlamKA, FastSlam1, FastSlam2KA, EnsKFSlamUA
from functions import h, H, f, F, H_wrt_l, h_inv, F_slam
from util import generate_simulation, generate_covariance_matrices, generate_large_state, comp_average_landmark_error, append_to_file, most_common
from viz import plot_final_slam_result, visualize_position_error_history
from oct2py import octave
from time import time
from copy import deepcopy
from scipy.linalg import inv, det, norm
from measurements import get_coordinate_from_range_bearing, mean_angles, associate_measurements, permute_measurements, get_range_bearing_measurements
from math import degrees
from numpy.linalg import norm as eucl_norm

np.set_printoptions(precision=3, suppress=True)

# Initialize the covariance matrices
ro_x = .5
ro_y = .5
ro_theta = 0.01

ro_range = .5
ro_bearing = 0.01

init_ro_x = 10
init_ro_y = 10
init_ro_theta = 0

Q, R, init_cov = generate_covariance_matrices(ro_x, ro_y, ro_theta, ro_range, ro_bearing, init_ro_x, init_ro_y, init_ro_theta)

Q_inflation = [2.5, 2.5, 2.5]
R_inflation = [2.5, 2.5]

# initialize additional arguments for the transition functions
(w, delta_t) = (1, 1)

# set up world
x_min = 0
x_max = 100

# initialize additional information
number_of_steps = 50
number_of_ensemble_members = 75
num_landmarks = 20

threshold = 40

plot_result, plot_prediction, plot_update = True, True, True
do_exkf, do_ens = False, True

folder = "../output/slam/association_test/"

start_time = time()

world_size, num_landmarks, landmarks, position_at_i, measurements_at_i, u_at_i = \
	generate_simulation(num_landmarks, x_min, x_max, orig_state=[5, 50, 0], threshold=threshold,
						number_of_steps=number_of_steps, f=f, Q=Q, R=R, step_size=2,
						additional_args=(w, delta_t), dim_state=3, go_straight=False, do_test=False, do_circle=False, asso_test=False)


# set up the Extended Kalman filter to use
large_dim = 3 + 2 * num_landmarks
large_state, large_init_cov = generate_large_state(position_at_i[0], num_landmarks, init_cov)

ExKFSlam = ExKFSlamKA(large_state, np.eye(large_dim), dim_state=len(large_state), dim_z=2,
					num_landmarks=num_landmarks, Q=Q, R=R)

# set up the Ensemble Kalman filter to use
# EnsKFSlam = EnsKFSlamKA(state=large_state,
# 						N=number_of_ensemble_members,
# 						dim_state=len(large_state),
# 						num_landmarks=num_landmarks,
# 						dim_z=2,
# 						Q=Q*Q_inflation,
# 						R=R*R_inflation,
# 						init_cov=large_init_cov)
EnsKFSlam = EnsKFSlamKA(state=large_state,
						N=number_of_ensemble_members,
						dim_state=len(large_state),
						num_landmarks=num_landmarks,
						dim_z=2,
						Q=Q*Q_inflation,
						R=R*R_inflation,
						init_cov=None)

EnsKFSlam_UA = EnsKFSlamUA(state=np.asarray(position_at_i[0]),
						N=number_of_ensemble_members,
						dim_state=3,
						num_landmarks=0,
						dim_z=2,
						Q=Q*Q_inflation,
						R=R*R_inflation,
						init_cov=None)

ex_prediction = [ExKFSlam.state]
ens_prediction = [EnsKFSlam.state]

###############################################################################################################

kk = 20

for i in range(1, number_of_steps+1):

	# print("--> Doing step %d" % i)

	if i % 25 == 0:
		print("--> Doing step %d" % i)
	u = u_at_i[i]
	z = measurements_at_i[i]

	# ###########################################################################################################
	if do_exkf:
		ExKFSlam.predict(f, F_slam, u, f_args=(w, delta_t), F_args=(w, delta_t))
		if plot_prediction:
			plot_final_slam_result("Progress for localization using an extended Kalman filter",
								folder + "ex_progress/step_%d_pred.png" % i, position_at_i[:i + 1],
								ex_prediction[:i] + [ExKFSlam.state],
								landmarks, x_min, x_max, None, None, do_particles=False, do_ensemble=False,
								path=True,
								show=False, save=True, slam=True)
		for j in range(len(z)):
			mes, according_landmark = z[j], landmarks[j]
			if mes is None:
				continue
			ExKFSlam.update(mes, j)

		ex_prediction += [ExKFSlam.state]

		if plot_update:
			plot_final_slam_result("Progress for localization using an extended Kalman filter",
								folder + "ex_progress/step_%d_up.png" % i, position_at_i[:i + 1],
								ex_prediction[:i + 1],
								landmarks, x_min, x_max, None, None, do_particles=False, do_ensemble=False,
								path=True,
								show=False, save=True, slam=False)
	# ###########################################################################################################
	if do_ens:
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print("--> Doing step %d" % i)
		# print("True position\n", position_at_i[i])

		EnsKFSlam_UA.predict(f, u, f_args=(w, delta_t), landmark_white_noise=0.0)

		# print("After predict\n", EnsKFSlam.state[:3])
		# print("After predict\n", EnsKFSlam_UA.state[:3])

		if plot_prediction:
			plot_final_slam_result("Progress for localization using an ensemble Kalman filter",
								folder + "ens_progress/step_%d_pred.png" % i, position_at_i[:i + 1],
								ens_prediction[:i] + [EnsKFSlam_UA.state],
								landmarks, x_min, x_max, None, EnsKFSlam_UA.ensemble, do_particles=False,
								do_ensemble=True, path=True, show=False, save=True, slam=False)

		# print("Landmarks: ", landmarks)
		if i == 1:  # current_state, landmarks, R, do_noise=True, threshold=None, dim_z=2
			z_noise = get_range_bearing_measurements(position_at_i[i], landmarks + [[30, 50]], R, do_noise=True, threshold=threshold)
			z = z_noise
		z_p, inds = permute_measurements([meas for meas in z if meas is not None])

		print("Got measurements: ", [meas for meas in z_p if meas is not None])
		print(".. .belonging to: ", [get_coordinate_from_range_bearing(meas, position_at_i[i]) for meas in z_p if meas is not None])
		print(EnsKFSlam_UA.state)

		EnsKFSlam_UA.full_update(z_p, h, H, threshold=threshold)

		# print(inds)
		# print("After update\n", EnsKFSlam.state[:3])
		# print("After update\n", EnsKFSlam_UA.state)
		# print("........................")

		ens_prediction += [EnsKFSlam_UA.state]

		if plot_update:
			plot_final_slam_result("Progress for localization using an ensemble Kalman filter",
								folder + "ens_progress/step_%d_up.png" % i, position_at_i[:i + 1],
								ens_prediction[:i + 1],
								landmarks, x_min, x_max, None, EnsKFSlam_UA.ensemble, do_particles=False,
								do_ensemble=True,
								path=True, show=False, save=True, slam=True, association=True)

		if eucl_norm(EnsKFSlam_UA.state[:2] - position_at_i[i][:2]) > 4.0:
			raise Exception

###############################################################################################################

if plot_result and do_exkf:
	plot_final_slam_result("Performance of ExSlam", folder + "exkf_slam.pdf",
						position_at_i, ex_prediction, landmarks, x_min, x_max, None, None, do_particles=False,
						do_ensemble=False, path=True, show=False, save=True, slam=True)
if plot_result and do_ens:
	plot_final_slam_result("Performance of EnsSlam with Data Association", folder + "enskf_slam.pdf",
						position_at_i, ens_prediction, landmarks, x_min, x_max, None, None, do_particles=False,
						do_ensemble=False, path=True, show=False, save=True, slam=True, association=True)
