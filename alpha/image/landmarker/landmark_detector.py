import cv2
import json
import os
from pathlib import Path
import numpy as np


img_folder = Path(os.path.dirname(os.getcwd())) / "preprocessing/data_raw"


def landmark_algorithim(img):
	# convert to grayscale
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# applying gaussian blur
	value = (35, 35)
	blurred = cv2.GaussianBlur(grey, value, 0)

	# thresholdin: Otsu's Binarization method
	_, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	# find contour with max area
	cnt = max(contours, key=lambda x: cv2.contourArea(x))

	# create bounding rectangle around the contour
	# x, y, w, h = cv2.boundingRect(cnt)
	# cv2.rectangle(img, (x - 10, y - 10), (x + w, y + h), (0, 225, 255), 0)

	# finding convex hull
	hull = cv2.convexHull(cnt)

	# drawing contours
	drawing = np.zeros(img.shape, np.uint8)
	cv2.drawContours(img, [cnt], 0, (0, 255, 0), 0)
	cv2.drawContours(img, [hull], 0, (0, 0, 255), 0)

	# determine the most extreme points along the contour
	# extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
	# extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
	extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
	# extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])

	landmarks.append(extTop)

	# cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
	# cv2.circle(img, extRight, 8, (0, 255, 0), -1)
	cv2.circle(img, extTop, 8, (255, 0, 0), -1)
	# cv2.circle(img, extBot, 8, (255, 255, 0), -1)
	return landmarks


# iterate over all images in img_folder
for img_name in os.listdir(img_folder):

	landmarks = []

	img = cv2.imread(str(img_folder / img_name))
	landmarks = landmark_algorithim(img)
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
