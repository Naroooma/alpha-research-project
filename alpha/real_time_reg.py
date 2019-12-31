import cv2
import numpy as np
from hand_segmentation import skin_threshold, reg_threshold
from landmark_finder import pointer_pos
from HCI import cursor

cap = cv2.VideoCapture(0)
cap.open


while(cap.isOpened()):
	# read image
	ret, img = cap.read()
	pressed = cv2.waitKey(1)

	# WORKING
	globalt = reg_threshold.YCrCb_thresh_2(img)
	# remove faces from threshold
	faces = reg_threshold.face_extract(img)
	for (x, y, w, h) in faces:
		cv2.rectangle(globalt, (x - 10, y), (x + w, y + h + 100), (0, 0, 0), -1)

	# find landmark
	try:
		# curx, cury = pointer_pos.top_pos(img, globalt)
		curx, cury = pointer_pos.far_pos(img, globalt)
		# move cursor using landmark
		cursor.move(curx, cury, cap.get(3), cap.get(4))
	# when threshold is empty
	except (ValueError, ZeroDivisionError):
		pass

	cv2.imshow('globalt', globalt)
	cv2.imshow('img', img)

	if pressed == ord('q'):
		break

cv2.destroyAllWindows()
