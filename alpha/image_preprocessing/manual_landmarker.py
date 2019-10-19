import cv2
import json
import os
from pathlib import Path


# iterate over each image in data_raw directory
for img_name in os.listdir(Path('data')):

	landmarks = []

	# draw and add landmark to list
	def draw_circle(event, x, y, flags, param):
		global landmarks
		if event == cv2.EVENT_LBUTTONDBLCLK:
			cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
			landmarks.append((x, y))
			print(x, y)

	# open image
	img = cv2.imread('data/' + img_name, 10)
	cv2.imshow('image', img)
	cv2.namedWindow('image')
	cv2.setMouseCallback('image', draw_circle)

	# save landmark positions to unique file in data_marked directory
	while True:
		cv2.imshow('image', img)
		if cv2.waitKey(1) == ord('q'):
			# add landmarks to JSON file for specific image if landmarks were added
			if landmarks != []:
				with open(Path('truth/') / (img_name + '.json'), 'w', encoding='utf-8') as f:
					json.dump(landmarks, f, ensure_ascii=False, indent=4)
			break

	cv2.destroyAllWindows()
