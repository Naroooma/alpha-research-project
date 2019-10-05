import cv2
import json
import os
from pathlib import Path


# open json files
json_folder = Path(os.path.dirname(os.getcwd())) / "preprocessing/data_marked"

for json_file in os.listdir(json_folder):
	with open(json_folder / json_file) as f:
		json_info = json.load(f)
	# print landmarks for each image
	for landmarks_list in json_info.values():
		print(json_file)
		for landmark in landmarks_list:
			print(landmark)
