import json
import os
from pathlib import Path
import numpy as np


def image_L2(img, results_folder, truth_folder):
	# results JSON
	with open(results_folder / img) as f:
		results_list = np.array(json.load(f))

	# truth JSON
	with open(truth_folder / img) as f:
		truth_list = np.array(json.load(f))
		print(truth_list)

	# loss calculation
	loss = np.linalg.norm(results_list - truth_list)
	return loss


truth_folder = Path(os.getcwd()) / "truth/personal"
results_folder = Path(os.getcwd()) / "results/personal"
total_loss = 0

# iterate over each image in results_folder
for results_file in os.listdir(results_folder):
	total_loss += image_L2(results_file, results_folder, truth_folder)

# print average loss
print(total_loss / len(os.listdir(results_folder)))
