import os
import glob

# Script must be run in the images parent directory
# outisde "piezas_repo/" 
for idx, image in enumerate(glob.glob('./*/*/*')):
    name = image.split("/")[3]
    direct = image.split("/")[2]
    os.rename(
        r'' + image,
        r'' + image.replace(name, direct + "_" + str(idx) + ".jpg")
    )
