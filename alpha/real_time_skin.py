import cv2
import numpy as np
from hand_segmentation import skin_threshold, reg_threshold
from landmark_finder import pointer_pos


cap = cv2.VideoCapture(0)
cap.open

global created_range
created_range = False

# create trackbars for color range expansion
cv2.namedWindow('expansion')
cv2.createTrackbar('BGR', 'expansion', 0, 500, skin_threshold.nothing)
cv2.createTrackbar('HSV', 'expansion', 0, 500, skin_threshold.nothing)
cv2.createTrackbar('YCrCb', 'expansion', 0, 500, skin_threshold.nothing)

while(cap.isOpened()):
	# read image
	ret, img = cap.read()
	pressed = cv2.waitKey(1)

	if created_range:
		# create YCrCb and HSV image for current frame
		YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
		HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		BGR_expansion = cv2.getTrackbarPos('BGR', "expansion")
		HSV_expansion = cv2.getTrackbarPos('HSV', "expansion")
		YCrCb_expansion = cv2.getTrackbarPos('YCrCb', "expansion")

		# thresholding for each colorspace
		thresh_HSV = cv2.inRange(HSV_img, (float(H.min()) - HSV_expansion, float(S.min()) - HSV_expansion, float(V.min()) - HSV_expansion),
			(float(H.max()) + HSV_expansion, float(S.max()) + HSV_expansion, float(V.max()) + HSV_expansion))

		thresh_YCrCb = cv2.inRange(YCrCb_img, (float(Y.min() - YCrCb_expansion), float(Cr.min()) - YCrCb_expansion, float(Cb.min()) - YCrCb_expansion),
			(float(Y.max()) + YCrCb_expansion, float(Cr.max()) + YCrCb_expansion, float(Cb.max()) + YCrCb_expansion))

		thresh_BGR = cv2.inRange(img, (float(B.min()) - BGR_expansion, float(G.min()) - BGR_expansion, float(R.min()) - BGR_expansion),
			(float(B.max()) + BGR_expansion, float(G.max()) + BGR_expansion, float(R.max()) + BGR_expansion))

		thresh_move = reg_threshold.movement_thresh(img)

		thresh_global = cv2.bitwise_and(thresh_YCrCb, thresh_HSV)
		blurred = cv2.GaussianBlur(thresh_global, (35, 35), 0)
		_, thresh_global = cv2.threshold(cv2.bitwise_not(blurred), 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		# blurred_global = cv2.GaussianBlur(thresh_global, (5, 5), 0)
		# _, thresh_global = cv2.threshold(blurred_global, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		# thresh_global = cv2.bitwise_not()

		cv2.imshow("move", thresh_move)
		cv2.imshow("HSV", thresh_HSV)
		cv2.imshow("YCrCb", thresh_YCrCb)
		cv2.imshow("BGR", thresh_BGR)
		cv2.imshow("global", thresh_global)
		# find contour with max area
		pointer_pos.far_pos(img, thresh_global)
		# pointer_pos.top_pos(img, thresh_global)

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
