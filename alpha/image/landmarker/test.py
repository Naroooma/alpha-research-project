import cv2
import numpy as np
import math


cap = cv2.VideoCapture(0)
cap.open

if not cap.isOpened():
	raise Exception("Could not open video device")
# Read picture. ret === True on success
ret, frame = cap.read()

while(cap.isOpened()):
	# read image
	ret, img = cap.read()


	# new = img - frame

	img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

	# skin color range for hsv color space

	YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))

	YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

	blur = cv2.GaussianBlur(YCrCb_mask, (3, 3), 0)

	_, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	# find contour with max area
	cnt = max(contours, key=lambda x: cv2.contourArea(x))

	# create bounding rectangle around the contour
	x, y, w, h = cv2.boundingRect(cnt)
	cv2.rectangle(img, (x - 10, y - 10), (x + w, y + h), (0, 225, 255), 0)

	# finding convex hull
	hull = cv2.convexHull(cnt)

	# drawing contours
	drawing = np.zeros(img.shape, np.uint8)
	cv2.drawContours(img, [cnt], 0, (0, 255, 0), 0)
	cv2.drawContours(img, [hull], 0, (0, 0, 255), 0)

	# determine the most extreme points along the contour
	extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
	extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
	extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
	extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])

	cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
	cv2.circle(img, extRight, 8, (0, 255, 0), -1)
	cv2.circle(img, extTop, 8, (255, 0, 0), -1)
	# cv2.circle(img, extBot, 8, (255, 255, 0), -1)

	cv2.imshow('image', img)
	# show results

	cv2.imshow("2_YCbCr.jpg", thresh1)
	# cv2.imshow("2_YCbCr.jpg", new)

	# background subtraction

	# split into 3 channels

	# thresholding

	# binarizaton

	# morphology

	# addition

	# morphology

	# face removal

	# canny edges segmentation

	# skin extraciton

	# morphology

	# gaussian filter

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))