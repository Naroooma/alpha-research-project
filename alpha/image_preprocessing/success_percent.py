import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


total_success = 0
total_img = 0
max_success_dist = 20
dists_1 = []
dists_2 = []
dists_3 = []
dists_4 = []


def image_success(img, dataset, dists):
	global total_img
	global total_success
	global max_success_dist

	# results JSON
	with open(Path(os.getcwd()) / "results" / dataset / img) as f:
		results_list = np.array(json.load(f))

	# truth JSON
	with open(Path(os.getcwd()) / "truth" / dataset / img) as f:
		truth_list = np.array(json.load(f))

	# L2 distance calculation
	dist = np.linalg.norm(results_list - truth_list)
	dists.append(dist)
	if dist <= max_success_dist:
		total_success += 1
	total_img += 1


def skin_image_success(img, dataset, algo, dists):
	global total_img
	global total_success
	global max_success_dist

	# results JSON
	with open(Path(os.getcwd()) / "results" / dataset / algo / img) as f:
		results_list = np.array(json.load(f))

	# truth JSON
	with open(Path(os.getcwd()) / "truth" / dataset / img) as f:
		truth_list = np.array(json.load(f))

	# L2 distance calculation
	dist = np.linalg.norm(results_list - truth_list)
	dists.append(dist)
	if dist <= max_success_dist:
		total_success += 1
	total_img += 1


def dist_histogram_all(dists):
	bins = int(1200 / 20)
	plt.xlabel("L2 Distance")
	plt.ylabel("Number of Images")
	plt.title("L2 distances of YCrCb_thresh_2")
	plt.hist(dists, bins=bins, rwidth=0.95,)
	# plt.legend(['calibration_data', 'calibration_white'])
	plt.legend(['alpha_data', 'kinect_leap', 'senz3d', 'alpha_white'])
	plt.show()


def dist_histogram_single(dists):
	bins = int(1200 / 20)
	plt.xlabel("L2 Distance")
	plt.ylabel("Number of Images")
	plt.title("L2 distances of YCrCb_thresh_2")
	plt.hist(dists, bins=bins, rwidth=0.95,)
	plt.show()


# # iterate over each image in results_folder
for results_file in os.listdir(Path(os.getcwd()) / "results" / "alpha_data"):
	image_success(results_file, "alpha_data", dists_1)

print(total_success, total_img)
print(str(total_success * 100 / total_img) + '%')

total_success = 0
total_img = 0

for results_file in os.listdir(Path(os.getcwd()) / "results" / "senz3d"):
	image_success(results_file, "senz3d", dists_3)

print(total_success, total_img)
print(str(total_success * 100 / total_img) + '%')

total_success = 0
total_img = 0

for results_file in os.listdir(Path(os.getcwd()) / "results" / "kinect_leap"):
	image_success(results_file, "kinect_leap", dists_2)

print(total_success, total_img)
print(str(total_success * 100 / total_img) + '%')

total_success = 0
total_img = 0

for results_file in os.listdir(Path(os.getcwd()) / "results" / "alpha_white"):
	image_success(results_file, "alpha_white", dists_4)

print(total_success, total_img)
print(str(total_success * 100 / total_img) + '%')

dist_histogram_all([dists_1, dists_2, dists_3, dists_4])

# -------- RESULTS FOR SKIN COLOR -------------

# for results_file in os.listdir(Path(os.getcwd()) / "results" / "calibration_white" / "HSV_skin_thresh"):
# 	skin_image_success(results_file, "calibration_white", "HSV_skin_thresh", dists_1)
# print(total_success, total_img)
# print(str(total_success * 100 / total_img) + '%')

# total_success = 0
# total_img = 0

# for results_file in os.listdir(Path(os.getcwd()) / "results" / "calibration_data" / "HSV_skin_thresh"):
# 	skin_image_success(results_file, "calibration_data", "HSV_skin_thresh", dists_2)

# print(total_success, total_img)
# print(str(total_success * 100 / total_img) + '%')

# dist_histogram_all([dists_1, dists_2])