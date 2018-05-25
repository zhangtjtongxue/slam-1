"""

This test will simulate a robot localization problem:
	* The world will be a quadratic world of size m, containing l landmarks in random positions
	* The robot will be placed in the middle of the world with a heading of 0 (facing east)
	* A series of movements will be generated. The controls specify a speed in which the robot moves and a turn angle
	* After moving, measurements containing (range,bearing) will be supplied to the robot.

Author:	C.Rauterberg

"""

import numpy as np
from filter import ExtendedKalmanFilter, EnsembleKalmanFilter, ParticleFilter
from functions import h, H, f, F
from util import generate_simulation, generate_covariance_matrices
from viz import plot_final_slam_result

# Initialize the covariance matrices
ro_x = .5
ro_y = .5
ro_theta = .01

ro_range = .5
ro_bearing = .01

init_ro_x = 25
init_ro_y = 25
init_ro_theta = 0

Q, R, init_cov = generate_covariance_matrices(ro_x, ro_y, ro_theta, ro_range, ro_bearing, init_ro_x, init_ro_y, init_ro_theta)

Q_inflation = [7.5, 7.5, 2.5]

# initialize additional arguments for the transition functions
(w, delta_t) = (1, 1)

# set up world
x_min = 0
x_max = 100

# initialize additional information
number_of_steps = 20
number_of_ensemble_members = 75
number_of_particles = 75
num_landmarks = 60

world_size, num_landmarks, landmarks, position_at_i, measurements_at_i, u_at_i = \
	generate_simulation(num_landmarks, x_min, x_max, orig_state=[5, 50, 0], threshold=None,
						number_of_steps=number_of_steps, f=f, Q=Q, R=R, step_size=3,
						additional_args=(w, delta_t), dim_state=3)

# set up the Extended Kalman filter to use
ExKF = ExtendedKalmanFilter(position_at_i[0], np.eye(3), dim_state=3, dim_z=2, Q=Q*Q_inflation, R=R)

# set up the Ensemble Kalman filter to use
EnsKF = EnsembleKalmanFilter(position_at_i[0], number_of_ensemble_members, dim_state=3, dim_z=2, Q=Q*Q_inflation, R=R, init_cov=init_cov)

# set up the particle filter to use
PF = ParticleFilter(number_of_particles, position_at_i[0], Q*Q_inflation, R, init_args=('gaussian', init_cov))

ex_prediction = [ExKF.state]
ens_prediction = [EnsKF.state]
pf_prediction = [PF.get_prediction()]

plot_result, plot_prediction, plot_update = True, False, False

do_ex, do_ens, do_pf = True, True, True

folder = "../output/localization/"

for i in range(1, number_of_steps+1):

	print("--> Doing step %d" % i)
	u = u_at_i[i]
	z = measurements_at_i[i]

	if do_ex:
		ExKF.predict(f, F, u, f_args=(w, delta_t), F_args=(w, delta_t))

		if plot_prediction:
			plot_final_slam_result("Progress for localization using an extended Kalman filter",
								folder + "ex_progress/step_%d_pred.png" % i, position_at_i[:i + 1],
								ex_prediction[:i] + [ExKF.state],
								landmarks, x_min, x_max, None, None, do_particles=False, do_ensemble=False,
								path=True,
								show=False, save=True, slam=False)

		for j in range(len(z)):
			mes, according_landmark = z[j], landmarks[j]
			if mes is None:
				continue
			ExKF.update(mes, h, H, h_args=according_landmark, H_args=according_landmark)

		ex_prediction += [ExKF.state]

		if plot_update:
			plot_final_slam_result("Progress for localization using an extended Kalman filter",
								folder + "ex_progress/step_%d_up.png" % i, position_at_i[:i + 1],
								ex_prediction[:i + 1],
								landmarks, x_min, x_max, None, None, do_particles=False, do_ensemble=False,
								path=True,
								show=False, save=True, slam=False)

	####################################################################################################################

	if do_ens:

		EnsKF.predict(f, u, f_args=(w, delta_t))

		if plot_prediction:
			plot_final_slam_result("Progress for localization using an ensemble Kalman filter",
								folder + "ens_progress/step_%d_pred.png" % i, position_at_i[:i + 1],
								ens_prediction[:i] + [EnsKF.state],
								landmarks, x_min, x_max, None, EnsKF.ensemble, do_particles=False, do_ensemble=True,
								path=True, show=False, save=True, slam=False)

		# after prediction, parse each measurement to update
		for j in range(len(z)):
			mes, according_landmark = z[j], landmarks[j]
			if mes is None:
				continue
			EnsKF.update(mes, h, h_args=according_landmark)

		ens_prediction += [EnsKF.state]

		# plot progress
		if plot_update:
			plot_final_slam_result("Progress for localization using an ensemble Kalman filter",
								folder + "ens_progress/step_%d_up.png" % i, position_at_i[:i + 1],
								ens_prediction[:i + 1],
								landmarks, x_min, x_max, None, EnsKF.ensemble, do_particles=False, do_ensemble=True,
								path=True, show=False, save=True, slam=False)

	####################################################################################################################

	if do_pf:

		PF.predict(u, f, f_args=(w, delta_t))

		# plot progress
		if plot_prediction:
			plot_final_slam_result("Progress for localization using an extended Kalman filter",
								folder + "pf_progress/step_%d_pred.png" % i, position_at_i[:i + 1], pf_prediction[:i] + [PF.get_prediction()],
								landmarks, x_min, x_max, PF.particles, None, do_particles=True, do_ensemble=False,
								path=True, show=False, save=True, slam=False)

		# after prediction, parse each measurement to update
		for j in range(len(z)):
			mes, according_landmark = z[j], landmarks[j]
			if mes is None:
				continue
			PF.update(mes, h, according_landmark)
		PF.resample()

		pf_prediction += [PF.get_prediction()]

		# plot progress
		if plot_update:
			plot_final_slam_result("Progress for localization using an extended Kalman filter",
								folder + "pf_progress/step_%d_up.png" % i, position_at_i[:i+1], pf_prediction[:i+1],
								landmarks, x_min, x_max, PF.particles, None, do_particles=True, do_ensemble=False,
								path=True, show=False, save=True, slam=False)

if plot_result and do_ex:
	plot_final_slam_result("Localization using an extended Kalman filter", folder + "exkf_localization.pdf",
						position_at_i, ex_prediction, landmarks, x_min, x_max, None, None, do_particles=False,
						do_ensemble=False, path=True, show=False, save=True, slam=False)
if plot_result and do_ens:
	plot_final_slam_result("Localization using an ensemble Kalman filter", folder + "enskf_localization.pdf",
						position_at_i, ens_prediction, landmarks, x_min, x_max, None, None, do_particles=False,
						do_ensemble=False, path=True, show=False, save=True, slam=False)
if plot_result and do_pf:
	plot_final_slam_result("Localization using a particle filter",  folder + "pf_localization.pdf",
						position_at_i, pf_prediction, landmarks, x_min, x_max, None, None, do_particles=False,
						do_ensemble=False, path=True, show=False, save=True, slam=False)
