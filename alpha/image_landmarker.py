import cv2
import json
import os
from pathlib import Path
import numpy as np
from hand_segmentation import reg_threshold
from landmark_finder import pointer_pos


# datasets = ['calibration_data', 'calibration_white']
datasets = ['alpha_data', 'senz3d', 'kinect_leap', 'alpha_white']

# iterate over datasets
for dataset in datasets:
	print(dataset)
	data_folder = Path(os.getcwd()) / "image_preprocessing/data" / dataset
	# results_folder = Path(os.getcwd()) / "image_preprocessing/results" / dataset / "YCrCb_thresh_6"
	results_folder = Path(os.getcwd()) / "image_preprocessing/results" / dataset

	# iterate over all images in img_folder
	for img_name in os.listdir(data_folder):
		print(img_name)
		landmarks = []

		img = cv2.imread(str(data_folder / img_name))

		thresh = reg_threshold.YCrCb_thresh_2(img)
		faces = reg_threshold.face_extract(img)
		for (x, y, w, h) in faces:
			cv2.rectangle(thresh, (x - 10, y), (x + w, y + h + 100), (0, 0, 0), -1)

		landmarks = pointer_pos.far_pos(img, thresh)

		cv2.circle(img, landmarks, 20, 1)
		# open image until q is pressed
		# while True:
		# 	cv2.imshow('image', img)
		# 	cv2.imshow('thresh', thresh)
		# 	if cv2.waitKey(0):
		# 		# fixes error not sure why
		# 		landmarks = np.ndarray.tolist(np.array(landmarks))
		# 		print(landmarks)
		# 		with open(results_folder / (img_name + '.json'), 'w', encoding='utf-8') as f:
		# 			json.dump(landmarks, f, ensure_ascii=False, indent=4)
		# 		break
		# cv2.destroyAllWindows()

		landmarks = np.ndarray.tolist(np.array(landmarks))
		with open(results_folder / (img_name + '.json'), 'w', encoding='utf-8') as f:
			json.dump(landmarks, f, ensure_ascii=False, indent=4)
