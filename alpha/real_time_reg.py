import cv2
import numpy as np
from hand_segmentation import skin_threshold, reg_threshold
from landmark_finder import pointer_pos


cap = cv2.VideoCapture(0)
cap.open


while(cap.isOpened()):
	# read image
	ret, img = cap.read()
	pressed = cv2.waitKey(1)

	gray = reg_threshold.gray_thresh(img)
	hsv = reg_threshold.HSV_thresh(img)
	ycrcb = reg_threshold.YCrCb_thresh(img)

	globalt = cv2.bitwise_and(gray, cv2.bitwise_not(ycrcb))
	faces = reg_threshold.face_extract(img)
	for (x, y, w, h) in faces:
		cv2.rectangle(globalt, (x - 10, y), (x + w, y + h + 100), (0, 0, 0), -1)

	# find contour with max area
	# pointer_pos.far_pos(img, globalt)
	pointer_pos.top_pos(img, globalt)
	cv2.imshow('globalt', globalt)
	cv2.imshow('hsv', hsv)
	cv2.imshow('ycrcb', ycrcb)
	cv2.imshow('gray', gray)
	cv2.imshow('img', img)

	if pressed == ord('q'):
		break

cv2.destroyAllWindows()
