import cv2
import numpy as np


def top_pos(img, thresh):
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

	return extTop


def far_pos(img, thresh):
	contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# find contour with max area
	try:
		max_cont = max(contours, key=lambda x: cv2.contourArea(x))
	except ValueError:
		return 0, 0
	# find centroid
	moment = cv2.moments(max_cont)
	cx = int(moment['m10'] / moment['m00'])
	cy = int(moment['m01'] / moment['m00'])

	hull1 = cv2.convexHull(max_cont, returnPoints=False)
	hull2 = cv2.convexHull(max_cont, returnPoints=True)

	# cv2.drawContours(img, [max_cont], 0, (0, 255, 0), 0)
	# cv2.drawContours(img, [hull2], 0, (0, 0, 255), 0)

	# find defects in max contour
	defects = cv2.convexityDefects(max_cont, hull1)
	if defects is None:
		return 0, 0

	# find farthest point
	s = defects[:, 0][:, 0]
	x = np.array(max_cont[s][:, 0][:, 0], dtype=np.float)
	y = np.array(max_cont[s][:, 0][:, 1], dtype=np.float)
	xp = cv2.pow(cv2.subtract(x, cx), 2)
	yp = cv2.pow(cv2.subtract(y, cy), 2)

	dist = cv2.sqrt(cv2.add(xp, yp))
	dist_max_i = np.argmax(dist)

	if dist_max_i < len(s):
		far_defect = s[dist_max_i]
		far_point = tuple(max_cont[far_defect][0])

		cv2.circle(img, far_point, 5, (0, 0, 255), -1)
		# cv2.circle(img, (cx, cy), 8, (255, 0, 0), -1)

		return far_point
	return 0, 0