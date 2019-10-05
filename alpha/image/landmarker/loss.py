import json
import os
from pathlib import Path
import numpy as np


def image_L2(img, results_folder, truth_folder):
	# results JSON
	with open(results_folder / img) as f:
		results_list = np.array(json.load(f))[0]

	# truth JSON
	with open(truth_folder / img) as f:
		truth_list = np.array(json.load(f))[0]

	# loss calculation
	loss = np.dot(results_list - truth_list, results_list - truth_list)
	return loss


truth_folder = Path(os.path.dirname(os.getcwd())) / "preprocessing/truth"
results_folder = Path('results')
total_loss = 0

# iterate over each image in results_folder
for results_file in os.listdir(results_folder):
	print(results_file)
	total_loss += image_L2(results_file, results_folder, truth_folder)

# print average loss
print(total_loss / len(os.listdir(results_folder)))
