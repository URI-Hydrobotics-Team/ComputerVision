# Script used for extract frames from one or more videos
# Run with python video_parser.py [video_path1] [video_path2] [notification_num]
# Or python3 video_parser.py [video_path1] [video_path2] [notification_num]

import cv2 as cv
import os
import sys

def parser(*args, notification_num):
	"""
	This function parse the video files and store each frames into a new folder

	Args:
		notification_num (int): How much frame processed before giving an update about it in terminal
		*args: One or more video to be processed
	"""

	# Loop through all the video paths
	for path in args:
		video = cv.VideoCapture(path)
		count = 0
		
		# Generate folder to contain the extracted frames
		video_name = os.path.splitext(os.path.basename(path))[0]
		folder = f'{os.getcwd()}/{video_name}'
		os.makedirs(folder, exist_ok=True)

		while video.isOpened():
			# Read frame
			ready, frame = video.read()

			if not ready:
				print(f'{count} frames created')
				break
			
			count += 1
			# Write the frame to the folder
			cv.imwrite(os.path.join(folder, f'{video_name}_frame{count}.jpg'), frame)

			# Send notification
			if count % notification_num == 0:
				print(f'{count} frames extracted')
			
		print('\n')

def main():
	if len(sys.argv) < 2:
		print("please pass in a video path")
		return

	*video_paths, notification_num = sys.argv[1:]

	parser(*video_paths, notification_num = int(notification_num))

if __name__ == "__main__":
	main()

