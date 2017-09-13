import os

def rename_imgs(dir):
	for filename in os.listdir(dir):
		if(filename.find('-') == -1):
			name = filename[0:filename.find('.')]
			print(name);
			ext = filename[filename.find('.'):]
			print(ext);
			os.rename(dir + filename, dir + name + '-0' + ext);

# rename_imgs('./roads_temp/');


#移除一定比例的图片
def remove(number, dir):
	i = 1;
	count = 0;
	for filename in os.listdir(dir):
		if i % number == 0:
			count += 1;
			os.remove(dir + filename);
			print(count, filename)
		i += 1;
# remove(2, './roads_temp/');