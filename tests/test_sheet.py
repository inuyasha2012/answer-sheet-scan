from sheet import get_answer_from_sheet
import cv2

base_img = cv2.imread('img/test.png')
get_answer_from_sheet(base_img)