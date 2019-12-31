import cv2
import numpy as np
import math
import sys
import os


fgbg = cv2.createBackgroundSubtractorMOG2()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def face_extract(img):
	faces = face_cascade.detectMultiScale(img, 1.1, 4)
	fs = []
	for (x, y, w, h) in faces:
		fs.append((x, y, w, h))
	return faces


def movement_thresh(img):
	fgmask = fgbg.apply(img)
	blurred = cv2.GaussianBlur(fgmask, (5, 5), 0)
	_, fgmask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	return cv2.bitwise_not(fgmask)


def binary_thresh(img):
	# convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# applying gaussian blur
	value = (35, 35)
	blurred = cv2.GaussianBlur(gray, value, 0)

	_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
	return cv2.bitwise_not(thresh)


def otsu_thresh(img):
	# convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# applying gaussian blur
	value = (35, 35)
	blurred = cv2.GaussianBlur(gray, value, 0)

	# thresholdin: Otsu's Binarization method
	_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	return thresh


def YCrCb_thresh_1(img):
	value = (35, 35)
	img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	# YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 90), (255, 200, 200))

	YCrCb_mask = cv2.inRange(img_YCrCb, (0, 133, 77), (255, 173, 127))
	return YCrCb_mask


def YCrCb_thresh_2(img):
	thresh = cv2.bitwise_and(YCrCb_thresh_1(img), binary_thresh(img))
	return thresh


def YCrCb_thresh_3(img):
	thresh = cv2.bitwise_and(YCrCb_thresh_1(img), otsu_thresh(img))
	return thresh


def YCrCb_thresh_4(img):
	value = (35, 35)
	img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	# YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 90), (255, 200, 200))
	# home
	YCrCb_mask = cv2.inRange(img_YCrCb, (0, 131, 110), (255, 157, 135))
	return YCrCb_mask


def YCrCb_thresh_5(img):
	thresh = cv2.bitwise_and(YCrCb_thresh_4(img), binary_thresh(img))
	return thresh


def YCrCb_thresh_6(img):
	thresh = cv2.bitwise_and(YCrCb_thresh_4(img), otsu_thresh(img))
	return thresh


def HSV_thresh_1(img):
	img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
	return HSV_mask


def HSV_thresh_2(img):
	thresh = cv2.bitwise_and(HSV_thresh_1(img), binary_thresh(img))
	return thresh


def HSV_thresh_3(img):
	thresh = cv2.bitwise_and(HSV_thresh_1(img), otsu_thresh(img))
	return thresh


def HSV_YCrCb_1(img):
	value = (35, 35)

	HSV_mask = HSV_thresh_2(img)
	HSV_blur = cv2.GaussianBlur(HSV_mask, value, 0)
	YCrCb_mask = YCrCb_thresh_6(img)
	YCrCb_blur = cv2.GaussianBlur(YCrCb_mask, value, 0)
	thresh = cv2.bitwise_and(HSV_blur, YCrCb_blur)

	blurred = cv2.GaussianBlur(thresh, value, 0)

	# thresholdin: Otsu's Binarization method
	_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	return cv2.bitwise_not(thresh)


def HSV_YCrCb_2(img):
	value = (35, 35)

	HSV_mask = HSV_thresh_2(img)
	HSV_blur = cv2.GaussianBlur(HSV_mask, value, 0)
	YCrCb_mask = YCrCb_thresh_2(img)
	YCrCb_blur = cv2.GaussianBlur(YCrCb_mask, value, 0)
	thresh = cv2.bitwise_and(HSV_blur, YCrCb_blur)

	blurred = cv2.GaussianBlur(thresh, value, 0)

	# thresholdin: Otsu's Binarization method
	_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	return cv2.bitwise_not(thresh)
