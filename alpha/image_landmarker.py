import cv2
import json
import os
from pathlib import Path
import numpy as np
from hand_segmentation import reg_threshold
from landmark_finder import pointer_pos


data_folder = Path(os.getcwd()) / "image_preprocessing/data"
results_folder = Path(os.getcwd()) / "image_preprocessing/results"

# iterate over all images in img_folder
for img_name in os.listdir(data_folder):

	landmarks = []

	img = cv2.imread(str(data_folder / img_name))

	gray = reg_threshold.gray_thresh(img)
	hsv = reg_threshold.HSV_thresh(img)
	ycrcb = reg_threshold.YCrCb_thresh(img)

	globalt = cv2.bitwise_and(gray, cv2.bitwise_not(ycrcb))
	faces = reg_threshold.face_extract(img)
	for (x, y, w, h) in faces:
		cv2.rectangle(globalt, (x - 10, y), (x + w, y + h + 100), (0, 0, 0), -1)

	landmarks = pointer_pos.top_pos(img, globalt)
	# open image until q is pressed
	while True:
		cv2.imshow('image', img)
		cv2.imshow('thresh', globalt)
		if cv2.waitKey(1) == ord('q'):

			# fixes error not sure why
			landmarks = np.ndarray.tolist(np.array(landmarks))

			with open(results_folder / (img_name + '.json'), 'w', encoding='utf-8') as f:
				json.dump(landmarks, f, ensure_ascii=False, indent=4)
			break

	cv2.destroyAllWindows()
