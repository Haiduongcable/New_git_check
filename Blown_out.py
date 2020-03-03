import cv2
import numpy as np
import json
import imutils
import os
'''
INPUT: PATH JSON, PATH IMG DIR
OUTPUT: TRUE, FALSE
NOTE: MY 'imagePath' is " ../picked_img/'name_img'.jpg ".
IF YOU HAVE THE DIFFERENCE 'imagePath',PLEASE FIX "path_img = path_img + str(img_dict['imagePath'])[3:]"!

'''
def release_rgb(img_rgb):
    lower = np.array([250,250,250], dtype = "uint8")
    upper = np.array([255,255,255], dtype = "uint8")
    mask = cv2.inRange(img_rgb, lower, upper)
    output = cv2.bitwise_and(img_rgb, img_rgb, mask = mask)
    return output

def convert_json_2_crop_image_custom(pathjson,path_img):
    with open(pathjson,'r') as myfile:
        data_in = myfile.read()
    img_dict = json.loads(data_in)
    path_img = path_img + str(img_dict['imagePath'])[3:]
    address_point = img_dict['shapes'][0]['points']
    label = img_dict['shapes'][0]['label']
    pts = np.array([(address_point[0][0],address_point[0][1]),
                    (address_point[1][0],address_point[1][1]),
                    (address_point[2][0],address_point[2][1]),
                    (address_point[3][0],address_point[3][1])],dtype = "float32")
    img_in = cv2.imread(path_img)
    return img_in ,pts ,label

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    rect[0] = pts[0]
    rect[2] = pts[2]
    rect[1] = pts[1]
    rect[3] = pts[3]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    	[0, 0],
    	[maxWidth - 1, 0],
    	[maxWidth - 1, maxHeight - 1],
    	[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image.copy(), M, (maxWidth, maxHeight))
    # return the warped image
    return cv2.resize(warped,(856,540))


def correct_illumination(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_space = img_hsv[:, :, 2].copy()
    # cv2.imwrite('e.png', v_space)
    background = cv2.morphologyEx(v_space, cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    background = background.astype(np.int32)
    v_hat_space = (v_space.astype(np.int32) - background + np.mean(background))

    v_hat_space[v_hat_space < 0] = 0
    v_hat_space[v_hat_space > 255] = 255

    v_hat_space = v_hat_space.astype(np.uint8)
    # cv2.imwrite('f.png', v_hat_space)
    v_hat_space = np.expand_dims(v_hat_space, axis=-1)

    img_concat = np.concatenate((img_hsv[:, :, :2], v_hat_space), axis=-1).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_concat, cv2.COLOR_HSV2BGR)
    return img_bgr

def find_brightness(img_trans):
    gray_img = cv2.cvtColor(release_rgb(img_trans), cv2.COLOR_BGR2GRAY)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_img, None, None, None, 8, cv2.CV_32S)

    areas = stats[1:, cv2.CC_STAT_AREA]

    result_test = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 70:
            result_test[labels == i + 1] = 255

    return result_test


def main(pathjson,pathimg):
    image_raw ,pts ,label = convert_json_2_crop_image_custom(pathjson,pathimg)
    img_trans = four_point_transform(image_raw, pts)
    out_img = correct_illumination(img_trans)
    if label == 'cmt':
        result = find_brightness(img_trans[260:])
    if label == 'back':
        result = find_brightness(img_trans)
    if (np.mean(result)) >= 0.55:
        return False ,out_img
    else:
        return True ,out_img
def nonjson_custom_main(img):
    img_in = cv2.resize(img,(856,540))
    out_img = correct_illumination(img_in[:,260:,:])
    result = find_brightness(img_in[:,260:,:])
    if (np.mean(result)) >= 0.75:
        return False ,out_img, result,img_in
    else:
        return True ,out_img, result,img_in


pathjson = '/home/duongnh/Downloads/Image_wrong/img_3.json'
pathimg = '/home/duongnh/Downloads/Image_wrong/'
