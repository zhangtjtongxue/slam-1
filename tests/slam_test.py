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
from slam import ExKFSlamKA, EnsKFSlamKA, FastSlam1, FastSlam2KA
from functions import h, H, f, F, H_wrt_l, h_inv, F_slam
from util import generate_simulation, generate_covariance_matrices, generate_large_state, comp_average_landmark_error, append_to_file
from viz import plot_final_slam_result, visualize_position_error_history, plot_matrix_heatmap
from oct2py import octave
from time import time
from copy import deepcopy
from scipy.linalg import inv, det, norm
from measurements import get_coordinate_from_range_bearing, mean_angles


# np.set_printoptions(precision=3, suppress=True)
octave.addpath('../matlab_fastslam')
octave.addpath('../matlab_fastslam/fastslam2')

# Initialize the covariance matrices
ro_x = 0.1
ro_y = 0.1
ro_theta = 0.001

ro_range = 0.1
ro_bearing = 0.001

init_ro_x = 5
init_ro_y = 5
init_ro_theta = .1

Q_ekf, R, init_cov = generate_covariance_matrices(ro_x, ro_y, ro_theta, ro_range, ro_bearing, init_ro_x, init_ro_y, init_ro_theta)

Q_v = 1.0
Q_g = 0.01
Q = np.asarray([[Q_v, 0], [0, Q_g]])

Q_inflation_ekf = [2.5, 2.5, 2.5]
Q_inflation = [2.5, 2.5]
R_inflation = [2.5, 2.5]

# initialize additional arguments for the transition functions
(w, delta_t) = (1, 1)

# initialize additional information
number_of_steps = 700
number_of_ensemble_members = 100
number_of_particles = 100
num_landmarks = 100

threshold = 40

plot_result, plot_prediction, plot_update = True, False, False
do_exkf, do_ens, do_matlab_part = False, True, False

sim_type = "random" # "random", "circle"
folder = "../output/slam/"

number_of_runs = 1

for run in range(number_of_runs):

	print("Doing run %d" % run)

	start_time = time()

	meas_R = np.zeros((2, 2))

	if sim_type == "random":
		# set up world
		x_min = -100
		x_max = 100

		world_size, num_landmarks, landmarks, position_at_i, measurements_at_i, u_at_i = \
			generate_simulation(num_landmarks=num_landmarks,
								x_min=x_min,
								x_max=x_max,
								orig_state=[-95, 0, 0],
								threshold=threshold,
								number_of_steps=number_of_steps,
								f=f,
								Q=Q,
								R=R,
								step_size=2.5,
								additional_args=(w, delta_t),
								dim_state=3,
								go_straight=False,
								do_test=False,
								do_circle=False)
	elif sim_type == "circle":
		# set up world
		x_min = -100
		x_max = 100

		world_size, num_landmarks, landmarks, position_at_i, measurements_at_i, u_at_i = \
			generate_simulation(num_landmarks=num_landmarks,
								x_min=x_min,
								x_max=x_max,
								orig_state=[-95, 0, 0],
								threshold=threshold,
								number_of_steps=number_of_steps,
								f=f,
								Q=Q,
								R=R,
								step_size=2.5,
								additional_args=(w, delta_t),
								dim_state=3,
								go_straight=False,
								do_test=False,
								do_circle=True)
	else:
		# set up world
		x_min = 0
		x_max = 100

		world_size, num_landmarks, landmarks, position_at_i, measurements_at_i, u_at_i = \
			generate_simulation(num_landmarks=num_landmarks,
								x_min=x_min,
								x_max=x_max,
								orig_state=[5, 50, 0],
								threshold=None,
								number_of_steps=number_of_steps,
								f=f,
								Q=Q,
								R=R,
								step_size=2.5,
								additional_args=(w, delta_t),
								dim_state=3,
								go_straight=False,
								do_test=True,
								do_circle=False)

	# set up the Extended Kalman filter to use
	large_dim = 3 + 2 * num_landmarks
	large_state, large_init_cov = generate_large_state(position_at_i[0], num_landmarks, init_cov, init=None)

	ExKFSlam = ExKFSlamKA(state=large_state,
						sigma=np.eye(large_dim),
						dim_state=len(large_state),
						dim_z=2,
						num_landmarks=num_landmarks,
						Q=Q_ekf*Q_inflation_ekf,
						R=R*R_inflation)

	# set up the Ensemble Kalman filter to use
	EnsKFSlam = EnsKFSlamKA(state=large_state,
							N=number_of_ensemble_members,
							dim_state=len(large_state),
							num_landmarks=num_landmarks,
							dim_z=2,
							Q=Q*Q_inflation,
							R=R*R_inflation,
							init_cov=None)

	LinEnsKFSlam = EnsKFSlamKA(state=large_state,
							N=number_of_ensemble_members,
							dim_state=len(large_state),
							num_landmarks=num_landmarks,
							dim_z=2,
							Q=Q*Q_inflation,
							R=R*R_inflation,
							init_cov=None)

	ex_prediction = [ExKFSlam.state]
	ens_prediction = [EnsKFSlam.state]
	linens_prediction = [LinEnsKFSlam.state]

	###############################################################################################################

	for i in range(1, number_of_steps+1):

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

			EnsKFSlam.predict(f, u, f_args=(w, delta_t), landmark_white_noise=0.0)
			LinEnsKFSlam.predict(f, u, f_args=(w, delta_t), landmark_white_noise=0.0)

			if plot_prediction:
				plot_final_slam_result("Progress for localization using an ensemble Kalman filter",
									folder + "ens_progress/step_%d_pred.png" % i, position_at_i[:i + 1],
									ens_prediction[:i] + [EnsKFSlam.state],
									landmarks, x_min, x_max, None, EnsKFSlam.ensemble, do_particles=False,
									do_ensemble=True,
									path=True, show=False, save=True, slam=False)

			for j in range(len(z)):
				mes, according_landmark = z[j], landmarks[j]
				if mes is None:
					continue
				EnsKFSlam.update(mes, h, j, h_args=according_landmark, full_ensemble=True)
				LinEnsKFSlam.update(mes, h, j, h_args=according_landmark, full_ensemble=False)

			ens_prediction += [EnsKFSlam.state]
			linens_prediction += [LinEnsKFSlam.state]

			if plot_update:
				plot_final_slam_result("Progress for localization using an ensemble Kalman filter",
									folder + "ens_progress/step_%d_up.png" % i, position_at_i[:i + 1],
									ens_prediction[:i + 1],
									landmarks, x_min, x_max, None, EnsKFSlam.ensemble, do_particles=False,
									do_ensemble=True,
									path=True, show=False, save=True, slam=True)

	###############################################################################################################

	if plot_result and do_exkf:
		plot_final_slam_result("Performance of ExSlam", folder + "exkf_slam.pdf",
							position_at_i, ex_prediction, landmarks, x_min, x_max, None, None, do_particles=False,
							do_ensemble=False, path=True, show=False, save=True, slam=True)
	if plot_result and do_ens:
		plot_final_slam_result("Performance of EnsSlam", folder + "enskf_slam.pdf",
							position_at_i, ens_prediction, landmarks, x_min, x_max, None, None, do_particles=False,
							do_ensemble=False, path=True, show=False, save=True, slam=True)

	###############################################################################################################

	full_history = []
	xtrue = []

	if do_matlab_part:

		print("Parse to octave!")

		# Extract only the turning angles from control inputs
		alphas = np.array([u[1] for u in u_at_i[1:]]).T

		#############################################
		#
		# Matlab gives a list of particles back, the particles are generated using the following:
		#
		# 	p(i).w = 1 / np;
		# 	p(i).xv = [0;0;0];
		# 	p(i).Pv = zeros(3);
		# 	p(i).xf = [];
		# 	p(i).Pf = [];
		# 	p(i).da = [];
		#   p(i).asso = [];
		#
		#############################################

		# print("Giving over alphas: ", alphas)

		results = octave.fastslam2_sim(landmarks, alphas)

		# print("Received results from octave!")

		fastslam_m_particles = results[0]

		weights = [fastslam_m_particles[i][0] for i in range(len(fastslam_m_particles))]

		robot_positions = [fastslam_m_particles[i][1] for i in range(len(fastslam_m_particles))]

		rob_pos = np.average(robot_positions, axis=0, weights=weights)

		asso = fastslam_m_particles[0][6][0]
		xtrue = fastslam_m_particles[0][7]
		xtrue = np.asarray(xtrue).T

		# compute landmark predictions
		num_obs_landmarks = len(asso)

		final_landmark_predictions = []
		for i in range(num_obs_landmarks):
			landmark_predictions = []
			# Loop over all particles
			for j in range(len(fastslam_m_particles)):
				pred = [[fastslam_m_particles[j][3][0][i], fastslam_m_particles[j][3][1][i]]]
				landmark_predictions += pred
			final_landmark_predictions += [np.average(landmark_predictions, axis=0, weights=weights)]

		associated_landmark_predictions = [[0.0, 0.0] for _ in range(num_landmarks)]
		for kk in range(len(asso)):
			# Matlab does a f***ing 1 based index!!!
			associated_landmark_predictions[int(asso[kk]) - 1] = final_landmark_predictions[kk]

		history = np.loadtxt("../mats/prediction_history.mat")

		full_history = []

		for elem in history:
			full_history += [elem]

		final_state = list(rob_pos) + list(associated_landmark_predictions)
		full_history += [np.array([item for sublist in final_state for item in sublist])]

		print(full_history)

		if plot_result:
			plot_final_slam_result("Performance of FastSlam 2.0", folder + "matlab_pf2_slam.pdf",
							position_at_i, full_history, landmarks, x_min, x_max, None, None, do_particles=False,
							do_ensemble=False, path=True, show=False, save=True, slam=True)

	ex_position_errors, ex_landmark_errors = [], []
	ens_position_errors, ens_landmark_errors = [], []
	linens_position_errors, linens_landmark_errors = [], []
	matlab_position_errors, matlab_landmark_errors = [], None

	print()
	print("----")
	if do_exkf:
		for x in range(1, number_of_steps + 1):
			ex_position_errors += [norm(position_at_i[x][:2] - ex_prediction[x][:2])]
			ex_landmark_errors += [comp_average_landmark_error(landmarks, ex_prediction[x][3:])]
	if do_ens:
		for x in range(1, number_of_steps + 1):
			ens_position_errors += [norm(position_at_i[x][:2] - ens_prediction[x][:2])]
			ens_landmark_errors += [comp_average_landmark_error(landmarks, ens_prediction[x][3:])]
		for x in range(1, number_of_steps + 1):
			linens_position_errors += [norm(position_at_i[x][:2] - linens_prediction[x][:2])]
			linens_landmark_errors += [comp_average_landmark_error(landmarks, linens_prediction[x][3:])]
	if do_matlab_part:
		for x in range(0, number_of_steps):
			matlab_position_errors += [norm(xtrue[x][:2] - full_history[x][:2])]
		matlab_landmark_errors = comp_average_landmark_error(landmarks, full_history[-1][3:])

	print("----")
	print()
	print(Q)
	print(R)
	print("Number of ensemble members: ", number_of_ensemble_members)
	print("Number of landmarks: ", num_landmarks)
	print("Final position errors:")
	if do_exkf:
		print("EKF:", sum(ex_position_errors)/number_of_steps)
	if do_ens:
		print("EnKF:", sum(ens_position_errors)/number_of_steps)
		print("LinEnKF:", sum(linens_position_errors)/number_of_steps)
	if do_matlab_part:
		print("FastSlam:", sum(matlab_position_errors)/number_of_steps)
	print("Final landmark errors:")
	if do_exkf:
		print("EKF:", ex_landmark_errors[-1])
	if do_ens:
		print("EnKF:", ens_landmark_errors[-1])
		print("LinEnKF:", linens_landmark_errors[-1])
	if do_matlab_part:
		print("FastSlam:", matlab_landmark_errors)
