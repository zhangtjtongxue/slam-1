"""

This file contains all auxiliary functions used for plotting the localization and SLAM problems with robots.

Author:	C.Rauterberg

"""
import matplotlib.pyplot as plt
from math import *
from scipy.spatial.distance import euclidean as euclidiance_distance
from matplotlib.patches import Arc
from measurements import get_range_bearing_measurements
from matplotlib.patches import Ellipse

# Typical colors:
# 	true robot position: #00ff00
# 	predicted robot position: #6666ff
ROBOT_COLORS = ["#00ff00", "#6666ff"]
COLOR_TRUE_POSITION = 0
COLOR_PRED_POSITION = 1


def visualize_world(landmarks, x_min, x_max, ticks_stepsize=20):
	"""
	Plots everything there is to do about the world.
	:param landmarks: The true positions of all the landmarks
	:param x_min: Min. x-value in Euclidean plane
	:param x_max: Max. x-value in Euclidean plane
	:param ticks_stepsize: The stepsize for the ticks.
	:return:
	"""
	# Plot a grid to x_max visualize the world
	grid = [x_min, x_max, x_min, x_max]
	plt.axis(grid)
	plt.grid(b=True, which='major', color='0.75', linestyle='--')
	plt.xticks([i for i in range(x_min, x_max, ticks_stepsize)], fontsize=20)
	plt.yticks([i for i in range(x_min, x_max, ticks_stepsize)], fontsize=20)

	if landmarks is not None:
		rec = plt.Rectangle((landmarks[0][0] - 0.5, landmarks[0][1] - 0.5), 1., 1., facecolor='#D68910',
							edgecolor='#330000', label="Landmark")
		plt.gca().add_patch(rec)

		for i in range(1, len(landmarks)):
			# plot true position of landmark
			rec = plt.Rectangle((landmarks[i][0] - 0.5, landmarks[i][1] - 0.5), 1., 1., facecolor='#D68910',
								edgecolor='#330000')
			plt.gca().add_patch(rec)

	plt.xlim(x_min - 5, x_max + 5)
	plt.ylim(x_min - 5, x_max + 5)


def plot_robot_path(path, robot_color_index, label, robot_width=1):
	"""
	Plot a certain path of a robot. Can be true path, can be prediction.
	:param path: The path of the robot
	:param robot_color_index: The index for the color
	:param label: The label of the robot
	:param robot_width: The width of the robot. Defaults to 1
	"""
	r_c = ROBOT_COLORS[robot_color_index]
	circle = plt.Circle((path[0][0], path[0][1]), robot_width, facecolor=r_c, alpha=0.4, edgecolor='#0000cc',
						label=label)
	plt.gca().add_patch(circle)

	for elem in path:
		x = elem[0]
		y = elem[1]
		theta = elem[2]

		# robot's position
		circle = plt.Circle((x, y), robot_width, facecolor=r_c, alpha=0.4, edgecolor='#0000cc')
		plt.gca().add_patch(circle)

		# robot's orientation
		arrow = plt.Arrow(x, y, 2 * cos(theta), 2 * sin(theta), alpha=0.75, facecolor='#000000', edgecolor='#000000')
		plt.gca().add_patch(arrow)


def plot_landmark_predictions_victoria(landmark_predictions, s=25.0):
	L = int(len(landmark_predictions)/2)
	for i in range(L):
		prediction_x, prediction_y = landmark_predictions[2 * i], landmark_predictions[2 * i + 1]

		circle = plt.Circle((prediction_x, prediction_y), s, facecolor='#cc0000', edgecolor='#330000')
		plt.gca().add_patch(circle)


def plot_landmark_predictions(landmark_predictions, landmarks, eps=4.0, association=False):
	"""
	Plot the predictions of landmarks produced by algorithms
	:param landmark_predictions: The list of landmark predictions
	:param landmarks: The true positions of the landmarks
	"""
	circle = plt.Circle((landmark_predictions[0], landmark_predictions[1]), 1., facecolor='#cc0000',
						edgecolor='#330000', label="Predicted Landmark")
	plt.gca().add_patch(circle)

	for i in range(len(landmarks)):
		if 2 * i >= len(landmark_predictions):
			continue
		l_x, l_y = landmarks[i][0], landmarks[i][1]
		prediction_x, prediction_y = landmark_predictions[2 * i], landmark_predictions[2 * i + 1]

		# Skip not observed landmarks
		if abs(prediction_x) < eps and abs(prediction_y) < eps:
			continue

		circle = plt.Circle((prediction_x, prediction_y), 1., facecolor='#cc0000', edgecolor='#330000')
		plt.gca().add_patch(circle)

		if not association:
			plt.plot([l_x, prediction_x], [l_y, prediction_y], color='#000000', alpha=0.5)


def plot_victoria(title, filename, est_path, min_x, max_x, min_y, max_y):
	plt.figure(title, figsize=(20., 20.))
	plt.title(title, fontsize=30)

	plot_robot_path(est_path, COLOR_PRED_POSITION, label="Estimated Robot Position", robot_width=2)

	plt.xlim(min_x - 5, max_x + 5)
	plt.ylim(min_y - 5, max_y + 5)
	plt.grid('on')
	# plot_landmark_predictions_victoria(est_path[-1][3:])
	plt.savefig(filename, dpi=200)


def plot_final_slam_result(title, filename, hist, pred, landmarks, x_min, x_max, particles, ensemble, do_particles=False, do_ensemble=False, path=True, show=False, save=True, slam=True, association=False):
	"""
	Handle to plot the whole result of the slam algorithms
	:param title: The plot title
	:param filename: The filename to store under
	:param hist: The history of the true positions of the robot
	:param pred: The prediction of the algorithms
	:param landmarks: The positions of the landmarks
	:param x_min: Min. x-value in Euclidean plane
	:param x_max: Max. x-value in Euclidean plane
	:param particles: The list of particles
	:param ensemble: The list of ensemble members
	:param do_particles: If true, plot the particles
	:param do_ensemble: If true, plot ensemble members
	:param path: If true, plot the path of the robot and the prediction path
	:param show: If true, show the plot
	:param save: If true, store the plot under the given filename
	"""
	plt.figure(title, figsize=(20., 20.))
	plt.title(title, fontsize=30)

	if path:
		plot_robot_path(pred, COLOR_PRED_POSITION, label="Estimated Robot Position")
		plot_robot_path(hist, COLOR_TRUE_POSITION, label="True Robot Position")

	if slam:
		plot_landmark_predictions(pred[-1][3:], landmarks, association=association)
	visualize_world(landmarks, x_min, x_max)

	if do_particles:
		visualize_particles(particles)
	if do_ensemble:
		visualize_ensemble(ensemble)

	plt.grid('on')
	plt.legend(loc='upper right', prop={'size': 30})

	if show:
		plt.show()
	if save:
		plt.savefig(filename, dpi=200)
	plt.clf()


def visualize_particles(particles, robot_width=1):
	"""
	Visualize the particles. Plot robot part of a particle and each landmark in each particle
	:param particles: The list of particles
	:param robot_width: The width of the robot. Defaults to 1
	"""
	robot_color = "#8b24e5"
	if len(particles[0]) > 3:
		circle = plt.Circle((particles[0][0][0], particles[0][0][1]), robot_width, facecolor=robot_color, alpha=0.4,
							edgecolor='#0000cc', label="Particle")
	else:
		circle = plt.Circle((particles[0][0], particles[0][1]), robot_width, facecolor=robot_color, alpha=0.4, edgecolor='#0000cc',label="Particle")
	plt.gca().add_patch(circle)

	for elem in particles:
		if isinstance(elem, list) and len(elem[0]) == 3:
			x = elem[0][0]
			y = elem[0][1]
			theta = elem[0][2]
		else:
			x = elem[0]
			y = elem[1]
			theta = elem[2]

		# robot's position
		circle = plt.Circle((x, y), robot_width, facecolor=robot_color, alpha=0.2, edgecolor='#0000cc')
		plt.gca().add_patch(circle)

		# robot's orientation
		arrow = plt.Arrow(x, y, 2 * cos(theta), 2 * sin(theta), alpha=0.2, facecolor='#000000', edgecolor='#000000')
		plt.gca().add_patch(arrow)

		if isinstance(elem, list) and len(elem[0]) == 3:
			num = elem[1]
			for i in range(num):
				ind = 2 + 2 * i
				if elem[ind] is None:
					continue
				plt.scatter(x=elem[ind][0], y=elem[ind][1], s=100, color='k', alpha=.5)


def visualize_ensemble(ensemble, dim_robot_state=3):
	"""
	Plot each ensemble member
	:param ensemble: A list of ensemble members
	:param dim_robot_state: The dimension of the state containing only robot information
	"""
	N = len(ensemble)
	plt.scatter(x=ensemble[0][0], y=ensemble[0][1], s=100, color='k', alpha=.5, label="Ensemble Member")
	for i in range(N):
		plt.scatter(x=ensemble[i][0], y=ensemble[i][1], s=100, color='k', alpha=.5)
		for j in range(int(len(ensemble[i][dim_robot_state:])/2)):
			ind = dim_robot_state + 2 * j
			plt.scatter(x=ensemble[i][ind], y=ensemble[i][ind+1], s=100, color='k', alpha=.5)


def visualize_position_error_history(exkf_history, enskf_history, fast_history, title, filename):
	plt.clf()
	ax = plt.gca()
	plt.figure(title, figsize=(20., 20.))
	plt.title(title, fontsize=30)

	max_error = max([max(exkf_history), max(enskf_history), max(fast_history)])

	x_ticks = [_ for _ in range(len(exkf_history))]

	plt.plot(x_ticks, exkf_history, c='g', label="EKF-Slam")
	plt.plot(x_ticks, enskf_history, c='r', label="EnKF-Slam")
	plt.plot(x_ticks, fast_history, c='b', label="FastSlam2.0")
	plt.grid('on')
	plt.legend(loc="upper left", prop={'size': 30})

	plt.xticks([i for i in range(0, len(exkf_history), 10)], fontsize=20)
	plt.yticks([i for i in range(0, int(max_error + 3), 1)], fontsize=20)

	plt.xlabel("Number of Steps", fontsize=20)
	plt.ylabel("Error in Robot's Position", fontsize=20)

	plt.savefig(filename, dpi=200)


def plot_matrix_heatmap(mat, filename):
	plt.imshow(mat, cmap='autumn', interpolation='nearest')
	plt.colorbar()
	plt.savefig(filename, dpi=200)
	plt.clf()


