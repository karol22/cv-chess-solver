import os
import glob

for idx, image in enumerate(glob.glob('./*/*')):
    name = image.split("/")[2]
    direct = image.split("/")[1]
    os.rename(
        r'' + image,
        r'' + image.replace(name, direct + "_" + str(idx) + ".png")
    )
