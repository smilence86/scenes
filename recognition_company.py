import os
import cv2
import urllib.request as ur
import numpy as np

dir = "./roads_temp/";

def gather_roads(url):
	i = 35938;
	while True:
		stream = ur.urlopen(url);
		imgNp = np.array(bytearray(stream.read()), dtype=np.uint8);
		img = cv2.imdecode(imgNp, -1);

		# if(i % 2 == 0):
		filepath = dir + "road_" + str(i) + ".jpg";
		print(filepath);
		cv2.imwrite(filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100]);
		i += 1;

# gather_roads('http://172.16.22.178:4747/mjpegfeed');
# gather_roads('http://172.16.21.187:8080/shot.jpg');
# gather_roads('http://192.168.43.1:8080/shot.jpg');

def rename_imgs():
	for filename in os.listdir(dir):
		if(filename.find('-') == -1):
			name = filename[0:filename.find('.')]
			print(name);
			ext = filename[filename.find('.'):]
			print(ext);
			os.rename(dir + filename, dir + name + '-0' + ext);

rename_imgs();

def recognition(url):
	while True:
		stream = ur.urlopen(url);
		imgNp = np.array(bytearray(stream.read()), dtype=np.uint8);
		img = cv2.imdecode(imgNp, -1);
		cv2.namedWindow("detect_line", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty("detect_line", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow('detect_line', img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
