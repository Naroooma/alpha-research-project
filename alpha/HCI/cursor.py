import pyautogui


def move(x, y, camera_width, camera_height):
	# get screen reslotion
	monitor_width, monitor_height = pyautogui.size()
	absolute_x = (x / camera_width) * monitor_width
	absolute_y = (y / camera_height) * monitor_height
	pyautogui.moveTo(absolute_x, absolute_y, duration=0.001)