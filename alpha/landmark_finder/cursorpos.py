import cv2
import numpy as np
import sys
import os


def top_pos(thresh, img):
	contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
	cv2.circle(img, extBot, 8, (255, 255, 0), -1)

	return img