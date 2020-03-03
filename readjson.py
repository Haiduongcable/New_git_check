import json
import cv2
import numpy as np
import imutils

with open('/home/duongnh/Downloads/too_bright_6.json', 'r') as myfile:
    data=myfile.read()

image_dict = json.loads(data)
#print(str(image_dict['height']))

for i in image_dict:
    print(i)

#print(str(image_dict['categories']))
shapes = image_dict['shapes'][0]['points']
print(shapes)
#def convert_draw(json):
    #for i in range()
print(str(image_dict['imagePath']))

'''
def convert_json_2_crop_image(pathjson):
    with open(pathjson,'r') as myfile:
        data_in = myfile.read()
    img_dict = json.loads(data_in)
    path_img = '/home/duongnh/Documents/ID_Card/' + str(img_dict['imagePath'][3:])
    address_point = img_dict['shapes'][0]['points']
    img_in = cv2.imread(path_img)
    image = img_in.copy()
    mask = np.ones(image.shape, dtype=np.uint8)
    mask.fill(255)
    roi_corners = np.array([address_point], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 0)
    crop_img = cv2.bitwise_or(image, mask)

    return imutils.resize(crop_img,height = 600)
'''
'''
path = '/home/duongnh/Documents/ID_Card/json_label_identity/1.json'
img_in = convert_json_2_crop_image(path)
hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
#_, contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow('HEY',hsv)
cv2.waitKey(0)


'''

