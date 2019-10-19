import cv2
import numpy as np
import math


cap = cv2.VideoCapture(0)
cap.open

traverse_point = []
if not cap.isOpened():
	raise Exception("Could not open video device")
# Read picture. ret === True on success
ret, frame = cap.read()
fgbg = cv2.createBackgroundSubtractorKNN()


def nothing(x):
	print(x)


def centroid(max_contour):

    moment = cv2.moments(max_contour)

    if moment['m00'] != 0:

        cx = int(moment['m10'] / moment['m00'])

        cy = int(moment['m01'] / moment['m00'])

        return cx, cy

    else:

        return None



def farthest_point(defects, contour, centroid):

    if defects is not None and centroid is not None:

        s = defects[:, 0][:, 0]

        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)

        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)

        yp = cv2.pow(cv2.subtract(y, cy), 2)

        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):

            farthest_defect = s[dist_max_i]

            farthest_point = tuple(contour[farthest_defect][0])

            return farthest_point

        else:

            return None


def draw_circles(frame, traverse_point):

    if traverse_point is not None:

        for i in range(len(traverse_point)):

            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)



def manage_image_opr(frame, thresh1):

	contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# find contour with max area
	max_cont = max(contours, key=lambda x: cv2.contourArea(x))

	cnt_centroid = centroid(max_cont)
	cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

	if max_cont is not None:

		hull = cv2.convexHull(max_cont, returnPoints=False)
		defects = cv2.convexityDefects(max_cont, hull)
		far_point = farthest_point(defects, max_cont, cnt_centroid)

		print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
		cv2.circle(img, far_point, 5, (0, 0, 255), -1)
		cv2.circle(frame, cnt_centroid, 8, (255, 0, 0), -1)

		# if len(traverse_point) < 20:
		#     traverse_point.append(far_point)

		# else:
		#     traverse_point.pop(0)
		#     traverse_point.append(far_point)

		# draw_circles(frame, traverse_point)


while(cap.isOpened()):
	# read image
	ret, img = cap.read()

	fgmask = fgbg.apply(img)
	blurred = cv2.GaussianBlur(fgmask, (5, 5), 0)
	_, fgmask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	fgmask = cv2.bitwise_not(fgmask)

	manage_image_opr(img, fgmask)
	cv2.imshow("f", fgmask)
	cv2.imshow("img", img)

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

# YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))