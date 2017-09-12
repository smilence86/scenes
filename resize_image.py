from PIL import Image
import os

# src = './roads_128/';
# dest = './roads_128/'

src = './roads_temp/';
dest = './roads_128/'

# dest = './roads_256/'
# dest = './roads_300/'

list = os.listdir(src)
print(list)

basewidth = 128

for image in list:
    id_tag = image.find(".")
    name = image[0:id_tag]
    # print(name)
    ext = image[id_tag:]
    # print(ext)

    img = Image.open(src + image)
    if img.size[0] == 640:
        print(name)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        out = img.resize((basewidth, hsize))
        # out.show()
        out.save(dest + name + ext)

