import cv2
import json
import os
from pathlib import Path
import numpy as np


img_folder = Path(os.path.dirname(os.getcwd())) / "preprocessing/data_raw"


def landmarker(thresh, img):

	contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	# find contour with max area
	cnt = max(contours, key=lambda x: cv2.contourArea(x))

	# finding convex hull
	hull = cv2.convexHull(cnt)

	# create bounding rectangle around the contour
	x, y, w, h = cv2.boundingRect(cnt)
	cv2.rectangle(img, (x - 10, y - 10), (x + w, y + h), (0, 225, 255), 0)

	# drawing contours
	drawing = np.zeros(img.shape, np.uint8)
	cv2.drawContours(img, [cnt], 0, (0, 255, 0), 0)
	cv2.drawContours(img, [hull], 0, (0, 0, 255), 0)

	# determine the most extreme points along the contour
	extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])

	landmarks.append(extTop)
	cv2.circle(img, extTop, 8, (255, 0, 0), -1)

	cv2.imshow("thresh", thresh)
	cv2.imshow('image', img)
	# return landmarks

	return landmarks


def grey_algorithim(img):
	# convert to grayscale
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# applying gaussian blur
	value = (35, 35)
	blurred = cv2.GaussianBlur(grey, value, 0)

	# thresholdin: Otsu's Binarization method
	_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	return landmarker(thresh, img)


def YCrCb_grey_algorithim(img):
	# convert colorspaces
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

	# skin color range for YCrCb
	YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 110), (255, 157, 135))

	# thresholding for YCrCb mask
	value = (35, 35)
	blurred_YCrCb = cv2.GaussianBlur(YCrCb_mask, value, 0)
	_, thresh_YCrCb = cv2.threshold(blurred_YCrCb, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	# thresholding for greyscale
	blurred_grey = cv2.GaussianBlur(grey, value, 0)
	_, thresh_grey = cv2.threshold(blurred_grey, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	thresh_global = thresh_grey - thresh_YCrCb
	_, thresh_global = cv2.threshold(thresh_global, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	thresh_global = cv2.bitwise_not(thresh_global)

	return landmarker(thresh_global, img)


def YCrCb_algorithim(img):
	# convert colorspaces
	img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

	# skin color range for YCrCb
	YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 110), (255, 157, 135))
	YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

	# thresholding for YCrCb mask
	value = (35, 35)
	blurred = cv2.GaussianBlur(YCrCb_mask, value, 0)
	_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	thresh = cv2.bitwise_not(thresh)

	return landmarker(thresh, img)


def move_algorithim(img):
	return 

# iterate over all images in img_folder
for img_name in os.listdir(img_folder):

	landmarks = []

	img = cv2.imread(str(img_folder / img_name))
	landmarks = YCrCb_algorithim(img)
	# open image until q is pressed
	while True:
		cv2.imshow('image', img)
		if cv2.waitKey(1) == ord('q'):

			# fixes error not sure why
			landmarks = np.ndarray.tolist(np.array(landmarks))

			with open(Path('results/') / (img_name + '.json'), 'w', encoding='utf-8') as f:
				json.dump(landmarks, f, ensure_ascii=False, indent=4)
			break

	cv2.destroyAllWindows()
