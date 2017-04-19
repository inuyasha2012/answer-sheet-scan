# coding=utf-8
import cv2
from imutils import contours
from e import ContourCountError, ContourPerimeterSizeError, PolyNodeCountError
from settings import ANS_IMG_THRESHOLD, CNT_PERIMETER_THRESHOLD, CHOICE_IMG_THRESHOLD, ANS_IMG_DILATE_ITERATIONS, \
    ANS_IMG_ERODE_ITERATIONS, CHOICE_IMG_DILATE_ITERATIONS, CHOICE_IMG_ERODE_ITERATIONS, CHOICE_MAX_AREA, \
    CHOICE_CNT_COUNT, ANS_IMG_KERNEL, CHOICE_IMG_KERNEL
from utils import detect_cnt_again, get_init_process_img, get_bright_process_img, get_max_area_cnt, get_ans


def get_answer_from_sheet(base_img):

    # cv2.imshow('temp', base_img)
    # cv2.waitKey(0)

    # 灰度化然后进行边缘检测、二值化等等一系列处理
    img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    img = get_init_process_img(img)

    # cv2.imshow('temp', img)
    # cv2.waitKey(0)

    # 获取最大面积轮廓并和图片大小作比较，看轮廓周长大小判断是否是答题卡的轮廓
    cnt = get_max_area_cnt(img)
    cnt_perimeter = cv2.arcLength(cnt, True)
    base_img_perimeter = (base_img.shape[0] + base_img.shape[1]) * 2
    if not cnt_perimeter > CNT_PERIMETER_THRESHOLD * base_img_perimeter:
        raise ContourPerimeterSizeError

    # cv2.drawContours(base_img, [cnt], 0, (0, 255, 0), 1)
    # cv2.imshow('temp', base_img)
    # cv2.waitKey(0)

    # 计算多边形的顶点，并看是否是四个顶点
    poly_node_list = cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.1, True)
    if not poly_node_list.shape[0] == 4:
        raise PolyNodeCountError

    # 根据计算的多边形顶点继续处理图片，主要是是纠偏
    processed_img = detect_cnt_again(poly_node_list, base_img)

    # cv2.imshow('temp', processed_img)
    # cv2.waitKey(0)

    # 调整图片的亮度
    processed_img = get_bright_process_img(processed_img)

    # cv2.imshow('temp', processed_img)
    # cv2.waitKey(0)

    # 通过二值化和膨胀腐蚀获得填涂区域
    ret, ans_img = cv2.threshold(processed_img, ANS_IMG_THRESHOLD[0], ANS_IMG_THRESHOLD[1], cv2.THRESH_BINARY_INV)
    ans_img = cv2.dilate(ans_img, ANS_IMG_KERNEL, iterations=ANS_IMG_DILATE_ITERATIONS)
    ans_img = cv2.erode(ans_img, ANS_IMG_KERNEL, iterations=ANS_IMG_ERODE_ITERATIONS)
    ret, ans_img = cv2.threshold(ans_img, ANS_IMG_THRESHOLD[0], ANS_IMG_THRESHOLD[1], cv2.THRESH_BINARY_INV)

    # cv2.imshow('temp', ans_img)
    # cv2.waitKey(0)

    # 通过二值化和膨胀腐蚀获得选项框区域
    ret, choice_img = cv2.threshold(processed_img, CHOICE_IMG_THRESHOLD[0], CHOICE_IMG_THRESHOLD[1],
                                    cv2.THRESH_BINARY_INV)
    choice_img = cv2.dilate(choice_img, CHOICE_IMG_KERNEL, iterations=CHOICE_IMG_DILATE_ITERATIONS)
    choice_img = cv2.erode(choice_img, CHOICE_IMG_KERNEL, iterations=CHOICE_IMG_ERODE_ITERATIONS)

    # cv2.imshow('temp', choice_img)
    # cv2.waitKey(0)

    # 查找选项框以及前面题号的轮廓
    cnts, h = cv2.findContours(choice_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    question_cnts = []

    temp_ans_img = ans_img.copy()
    for i, c in enumerate(cnts):
        # 如果面积小于某值，则认为这个轮廓是选项框或题号
        if cv2.contourArea(c) < CHOICE_MAX_AREA:
            cv2.drawContours(temp_ans_img, cnts, i, (0, 0, 0), 1)
            question_cnts.append(c)

    # cv2.imshow('temp', temp_ans_img)
    # cv2.waitKey(0)

    # 如果轮廓小于特定值，重新扫描
    # TODO 运用统计分析排除垃圾轮廓
    if len(question_cnts) != CHOICE_CNT_COUNT:
        raise ContourCountError

    # 对轮廓之上而下的排序
    question_cnts = contours.sort_contours(question_cnts, method="top-to-bottom")[0]

    # 获得答案
    get_ans(ans_img, question_cnts)
