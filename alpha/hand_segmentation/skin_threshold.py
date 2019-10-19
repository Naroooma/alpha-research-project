import cv2
import numpy as np
import time
# import sys
# sys.path.append()

fgbg = cv2.createBackgroundSubtractorMOG2()


def draw_squares(img):
	# draw squares
	rows, cols, _ = img.shape
	global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

	hand_rect_one_x = np.array(
		[6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
			12 * rows / 20, 12 * rows / 20], dtype=np.uint32)
	hand_rect_one_y = np.array(
		[9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
			10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

	hand_rect_two_x = hand_rect_one_x + 10
	hand_rect_two_y = hand_rect_one_y + 10

	for i in range(9):

		cv2.rectangle(img, (hand_rect_one_y[i], hand_rect_one_x[i]), (hand_rect_two_y[i], hand_rect_two_x[i]), (0, 255, 0), 1)

	return img


def skin_values_HSV(img):
	# create HSV frame
	hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# find values from area of interest
	aoi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

	# extract colors from squares
	for i in range(9):
		aoi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10, hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

	# split color channels
	return cv2.split(aoi)


def gaus_thresh(img):
	# thresholding img with gaussian
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	thresh_gaus = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

	blurred_gaus = cv2.GaussianBlur(thresh_gaus, (5, 5), 0)
	_, thresh_gaus = cv2.threshold(blurred_gaus, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	return thresh_gaus


def skin_values_YCrCb(img):
	# create YCrCb frame
	YCrCb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

	# find values from area of interest
	aoi = np.zeros([90, 10, 3], dtype=YCrCb_frame.dtype)

	# extract colors from squares
	for i in range(9):
		aoi[i * 10: i * 10 + 10, 0: 10] = YCrCb_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10, hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

	# split color channels
	return cv2.split(aoi)


def skin_values_BGR(img):
	# find values from area of interset
	aoi = np.zeros([90, 10, 3], dtype=img.dtype)

	# extract colors from squares
	for i in range(9):
		aoi[i * 10: i * 10 + 10, 0: 10] = img[hand_rect_one_x[i]:hand_rect_one_x[i] + 10, hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

	# split color channels
	return cv2.split(aoi)


def movement_thresh(img):
	fgmask = fgbg.apply(img)
	blurred = cv2.GaussianBlur(fgmask, (5, 5), 0)
	_, fgmask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	return cv2.bitwise_not(fgmask)


def face_extraction(img):
	return


def nothing(x):
	pass
