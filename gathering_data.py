import cv2
import urllib.request as ur
import numpy as np

# dir = "./roads_temp/";

def gather_roads(url, dir):
	i = 44793;
	while True:
		stream = ur.urlopen(url);
		imgNp = np.array(bytearray(stream.read()), dtype=np.uint8);
		img = cv2.imdecode(imgNp, -1);

		# if(i % 2 == 0):
		filepath = dir + "road_" + str(i) + ".jpg";
		print(filepath);
		cv2.imwrite(filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100]);
		i += 1;

# gather_roads('http://172.16.22.178:4747/mjpegfeed', './roads_temp/');
# gather_roads('http://172.16.21.187:8080/shot.jpg', './roads_temp/');
# gather_roads('http://192.168.43.1:8080/shot.jpg', './roads_temp/');
gather_roads('http://172.20.10.2:8080/shot.jpg', './roads_temp/');