import cv2
import numpy as np
import json
import imutils
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from Blown_out import main,convert_json_2_crop_image_custom,four_point_transform, nonjson_custom_main
def imshow(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.show()
def images_show(images):
    fig = plt.figure(figsize=(15., 15.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, images):
        imgshow(ax, im)
    plt.show()


def imgshow(ax, img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    else:
        ax.imshow(img, cmap='gray')

'''
pathjson = '/home/duongnh/Documents/ID_Card/picked_img_label_identity/'
pathimg= '/home/duongnh/Documents/ID_Card/'
os_dir_json = os.listdir('/home/duongnh/Documents/ID_Card/picked_img_label_identity')
img_trans_list = []
img = os_dir_json[5]
img_raw, pts, label = convert_json_2_crop_image_custom(pathjson  + str(img),pathimg)
img_correct = four_point_transform(img_raw,pts)
a = main(pathjson + str(img),pathimg)
imshow(img_correct)
print(a)
'''
list_img = []
dir = os.listdir('/home/duongnh/Documents/im_out')
for i in dir:
    img_in = cv2.imread('/home/duongnh/Documents/im_out/' + str(i))
    list_img.append(img_in)
print(len(list_img))
'''
for i in list_img:
    bool, out_img, result, img_o = nonjson_custom_main(i)
    print(bool)
    images_show([img_o,out_img])
'''
img_hsv = cv2.cvtColor(list_img[0],cv2.COLOR_BGR2HSV)
imshow(img_hsv[:,:,2])

#bool, out_img, result = nonjson_custom_main(img)

#print(bool)
#imshow(out_img)