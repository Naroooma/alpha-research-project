import cv2
import numpy as np
import math
import sys
import os


fgbg = cv2.createBackgroundSubtractorMOG2()


def movement_thresh(img):
	fgmask = fgbg.apply(img)
	blurred = cv2.GaussianBlur(fgmask, (5, 5), 0)
	_, fgmask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	return cv2.bitwise_not(fgmask)


def YCrCb_thresh(img):
	value = (35, 35)
	img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 90), (255, 200, 200))

	# home
	# YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 110), (255, 157, 135))

	blurred = cv2.GaussianBlur(YCrCb_mask, value, 0)
	_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	return thresh


def HSV_thresh(img):
	value = (35, 35)
	img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	HSV_mask = cv2.inRange(img_HSV, (0, 135, 90), (255, 200, 200))

	blurred = cv2.GaussianBlur(HSV_mask, value, 0)
	_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	return thresh


def gray_thresh(img):
	# convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# applying gaussian blur
	value = (35, 35)
	blurred = cv2.GaussianBlur(gray, value, 0)

	# thresholdin: Otsu's Binarization method
	_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	return landmarker(thresh, img)


def gaussian_thresh(img):
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

	blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
	_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	return thresh


def main():
	while(cap.isOpened()):
		# read image
		ret, img = cap.read()

		value = (35, 35)
		# convert to grayscale
		thresh_gaus = gaussian_thresh(img)
		thresh_HSV = HSV_mask(img)
		thresh_YCrCb = YCrCb_mask(img)

		thresh_global = cv2.bitwise_not(thresh_HSV) - thresh_YCrCb

		cv2.imshow("threshold1", cv2.bitwise_not(thresh_HSV))
		cv2.imshow("threshold2", thresh_YCrCb)
		cv2.imshow("thresh", thresh_global)
		cv2.imshow('image', img)
		pressed = cv2.waitKey(1)
		if pressed == ord('q'):
			break

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
