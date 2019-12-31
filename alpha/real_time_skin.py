import cv2
import json
import os
from pathlib import Path
import numpy as np
from hand_segmentation import skin_threshold, reg_threshold
from landmark_finder import pointer_pos


cap = cv2.VideoCapture(0)
cap.open

data_folder = Path(os.getcwd()) / "image_preprocessing/data" / "calibration_data"
results_folder = Path(os.getcwd()) / "image_preprocessing/results" / "calibration_data"

pictures_taken = 0
threshs = ['HSV_skin_thresh', 'YCrCb_skin_thresh', 'BGR_skin_thresh', 'HSV_BGR', 'HSV_YCrCb', 'YCrCb_BGR', 'all_thresh_binary', 'all_thresh_reg', 'all_thresh_otsu']

global created_range
created_range = False

while(cap.isOpened()):
	# read image
	ret, img = cap.read()
	pressed = cv2.waitKey(1)

	if created_range:

		# thresholding for each colorspace
		HSV_skin_thresh = skin_threshold.HSV_thresh(img, H, S, V)
		YCrCb_skin_thresh = skin_threshold.YCrCb_thresh(img, Y, Cr, Cb)
		BGR_skin_thresh = skin_threshold.BGR_thresh(img, B, G, R)

		# create thresholding combinations
		HSV_YCrCb = skin_threshold.HSV_YCrCb_thresh(HSV_skin_thresh, YCrCb_skin_thresh)
		HSV_BGR = skin_threshold.HSV_BGR_thresh(HSV_skin_thresh, BGR_skin_thresh)
		YCrCb_BGR = skin_threshold.YCrCb_BGR_thresh(YCrCb_skin_thresh, BGR_skin_thresh)
		all_thresh_reg = skin_threshold.all_thresh(HSV_YCrCb, BGR_skin_thresh)

		# Global Threshold + Otsu's Binarization
		all_thresh_otsu = skin_threshold.all_otsu(all_thresh_reg)
		all_thresh_binary = skin_threshold.all_binary(all_thresh_reg)

		# show threshs
		cv2.imshow("HSV", HSV_skin_thresh)
		cv2.imshow("YCrCb", YCrCb_skin_thresh)
		cv2.imshow("BGR", BGR_skin_thresh)
		cv2.imshow("global", all_thresh_reg)

		# pointer_pos.top_pos(img, all_thresh)
		if pictures_taken % 5 == 0 and pictures_taken / 5 <= 50 and pictures_taken != 0:
			# save photo into data folder
			cv2.imwrite(os.path.join(data_folder, (str(int(pictures_taken / 5)) + ".jpg")), img)

			# iterate over all thresholds
			for thresh in threshs:
				# convert thresh name to actual thresh
				exec("%s = %s" % ("thresh_var", thresh))

				# find landmarks based on threshold
				landmarks = pointer_pos.far_pos(img, thresh_var)
				landmarks = np.ndarray.tolist(np.array(landmarks))
				# save landmarks to corresponding json file for image
				with open(results_folder / str(thresh) / (str(int(pictures_taken / 5)) + '.jpg.json'), 'w', encoding='utf-8') as f:
					json.dump(landmarks, f, ensure_ascii=False, indent=4)

		print(pictures_taken)
		pictures_taken += 1

	else:
		# draw squares until ranges are created
		img = skin_threshold.draw_squares(img)

		if pressed == ord('w'):
			# when created_range == True, won't draw squares and won't create range
			created_range = True
			# extract color space values for each channel from area of interest
			H, S, V = skin_threshold.skin_values_HSV(img)
			Y, Cr, Cb = skin_threshold.skin_values_YCrCb(img)
			B, G, R = skin_threshold.skin_values_BGR(img)

	cv2.imshow('img', img)

	if pressed == ord('q'):
		break

cv2.destroyAllWindows()
