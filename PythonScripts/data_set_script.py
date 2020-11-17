import numpy as np
import cv2
import glob

from random import random
from alive_progress import alive_bar


def sp_noise(image):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    prob = random()/20 # Noise probability [.00 ~ .05]
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def rotate_n_noise_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return sp_noise(result)


def create_dataset():
    # Script must be run in the images parent directory
    # outisde "piezas_repo/" 
    #
    # piezas_repo/
    #  - torre_roja/
    #    - torre_fondo_blanco_roja.png
    #    - torre_fondo_blanco_roja.png
    #    - torre_fondo_negro_roja.png
    #    - piezas_repo/torre_roja/torre_fondo_blanco_roja_45_3_32.png
    #  - torre_azul
    #  - fondo_blanco
    #  - fondo_negro
    #  . . .
    images = glob.glob('./*/*/*.jpg')

    with alive_bar(32000) as bar: # 32,000 images to be processed
        for image in images:
            for i in range(1000):
                new_img_name = image.replace(
                    ".jpg",
                    "_{0}_.jpg".format(i)
                )

                cv2.imwrite(
                    new_img_name,
                    sp_noise(cv2.imread(image))
                )
                bar()


create_dataset()
