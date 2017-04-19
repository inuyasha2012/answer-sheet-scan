# coding=utf-8
import numpy as np
import cv2
from imutils import auto_canny, contours
from e import PolyNodeCountError
from settings import CHOICES, SHEET_AREA_MIN_RATIO, PROCESS_BRIGHT_COLS, PROCESS_BRIGHT_ROWS, BRIGHT_VALUE, \
    CHOICE_COL_COUNT, CHOICES_PER_QUE, WHITE_RATIO_PER_CHOICE, MAYBE_MULTI_CHOICE_THRESHOLD


def get_corner_node_list(poly_node_list):
    """
    获得多边形四个顶点的坐标
    :type poly_node_list: ndarray
    :return: tuple
    """
    center_y, center_x = (np.sum(poly_node_list, axis=0) / 4)[0]
    top_left = bottom_left = top_right = bottom_right = None
    for node in poly_node_list:
        x = node[0, 1]
        y = node[0, 0]
        if x < center_x and y < center_y:
            top_left = node
        elif x < center_x and y > center_y:
            bottom_left = node
        elif x > center_x and y < center_y:
            top_right = node
        elif x > center_x and y > center_y:
            bottom_right = node
    return top_left, bottom_left, top_right, bottom_right


def detect_cnt_again(poly, base_img):
    """
    继续检测已截取区域是否涵盖了答题卡区域
    :param poly: ndarray
    :param base_img: ndarray
    :return: ndarray
    """
    # 该多边形区域是否还包含答题卡区域的flag
    flag = False

    # 计算多边形四个顶点，并且截图，然后处理截取后的图片
    top_left, bottom_left, top_right, bottom_right = get_corner_node_list(poly)
    roi_img = get_roi_img(base_img, bottom_left, bottom_right, top_left, top_right)
    img = get_init_process_img(roi_img)

    # 获得面积最大的轮廓
    cnt = get_max_area_cnt(img)

    # 如果轮廓面积足够大，重新计算多边形四个顶点
    if cv2.contourArea(cnt) > roi_img.shape[0] * roi_img.shape[1] * SHEET_AREA_MIN_RATIO:
        flag = True
        poly = cv2.approxPolyDP(cnt, cv2.arcLength((cnt,), True) * 0.1, True)
        top_left, bottom_left, top_right, bottom_right = get_corner_node_list(poly)
        if not poly.shape[0] == 4:
            raise PolyNodeCountError

    # 多边形顶点和图片顶点，主要用于纠偏
    base_poly_nodes = np.float32([top_left[0], bottom_left[0], top_right[0], bottom_right[0]])
    base_nodes = np.float32([[0, 0],
                            [base_img.shape[1], 0],
                            [0, base_img.shape[0]],
                            [base_img.shape[1], base_img.shape[0]]])
    transmtx = cv2.getPerspectiveTransform(base_poly_nodes, base_nodes)

    if flag:
        img_warp = cv2.warpPerspective(roi_img, transmtx, (base_img.shape[1], base_img.shape[0]))
    else:
        img_warp = cv2.warpPerspective(base_img, transmtx, (base_img.shape[1], base_img.shape[0]))
    return img_warp


def get_init_process_img(roi_img):
    """
    对图片进行初始化处理，包括，梯度化，高斯模糊，二值化，腐蚀，膨胀和边缘检测
    :param roi_img: ndarray
    :return: ndarray
    """
    h = cv2.Sobel(roi_img, cv2.CV_32F, 0, 1, -1)
    v = cv2.Sobel(roi_img, cv2.CV_32F, 1, 0, -1)
    img = cv2.add(h, v)
    img = cv2.convertScaleAbs(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=2)
    img = auto_canny(img)
    return img


def get_roi_img(base_img, bottom_left, bottom_right, top_left, top_right):
    """
    截取合适的图片区域
    :param base_img: ndarray
    :param bottom_left: ndarray
    :param bottom_right: ndarray
    :param top_left: ndarray
    :param top_right: ndarray
    :return: ndarray
    """
    min_v = top_left[0, 1] if top_left[0, 1] < bottom_left[0, 1] else bottom_left[0, 1]
    max_v = top_right[0, 1] if top_right[0, 1] > bottom_right[0, 1] else bottom_right[0, 1]
    min_h = top_left[0, 0] if top_left[0, 0] < top_right[0, 0] else top_right[0, 0]
    max_h = bottom_left[0, 0] if bottom_left[0, 0] > bottom_right[0, 0] else bottom_right[0, 0]
    roi_img = base_img[min_v + 10:max_v - 10, min_h + 10:max_h - 10]
    return roi_img


def get_bright_process_img(img):
    """
    改变图片的亮度，方便二值化
    :param img: ndarray
    :return: ndarray
    """
    for y in range(PROCESS_BRIGHT_COLS):
        for x in range(PROCESS_BRIGHT_ROWS):
            col_low = 1.0 * img.shape[0] / PROCESS_BRIGHT_COLS * y
            col_high = 1.0 * img.shape[0] / PROCESS_BRIGHT_COLS * (y + 1)
            row_low = 1.0 * img.shape[1] / PROCESS_BRIGHT_ROWS * x
            row_high = 1.0 * img.shape[1] / PROCESS_BRIGHT_ROWS * (x + 1)
            roi = img[int(col_low):int(col_high), int(row_low): int(row_high)]
            mean = cv2.mean(roi)
            for each_roi in roi:
                for each_p in each_roi:
                    each_p += BRIGHT_VALUE - np.array(mean, dtype=np.uint8)[:3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def get_max_area_cnt(img):
    """
    获得图片里面最大面积的轮廓
    :param img: ndarray
    :return: ndarray
    """
    cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=lambda c: cv2.contourArea(c))
    return cnt


def get_ans(ans_img, question_cnts):
    # 选项个数加上题号
    interval = CHOICES_PER_QUE + 1
    for q, i in enumerate(np.arange(0, len(question_cnts), CHOICE_COL_COUNT)):
        # 从左到右为当前题目的气泡轮廓排序，然后初始化被涂画的气泡变量
        cnts = contours.sort_contours(question_cnts[i:i + CHOICE_COL_COUNT])[1]
        for k in range(3):
            print '======================================='
            percent_list = []
            for j, c in enumerate(cnts[1 + k * interval:interval + k * interval]):
                # 获得选项框的区域
                new = ans_img[c[1]:(c[1] + c[3]), c[0]:(c[0] + c[2])]
                # 计算白色像素个数和所占百分比
                white_count = np.count_nonzero(new)
                percent = white_count * 1.0 / new.size
                percent_list.append({'col': k + 1, 'row': q + 1, 'percent': percent, 'choice': CHOICES[j]})

            percent_list.sort(key=lambda x: x['percent'])
            choice_pos_n_ans = (percent_list[0]['row'], percent_list[0]['col'], percent_list[0]['choice'])
            choice_pos = (percent_list[0]['row'], percent_list[0]['col'])
            if percent_list[1]['percent'] < WHITE_RATIO_PER_CHOICE and \
                            abs(percent_list[1]['percent'] - percent_list[0]['percent']) < MAYBE_MULTI_CHOICE_THRESHOLD:
                print u'第%s排第%s列的作答：可能多涂了选项' % choice_pos
                print u"第%s排第%s列的作答：%s" % choice_pos_n_ans
            elif percent_list[0]['percent'] < WHITE_RATIO_PER_CHOICE:
                print u"第%s排第%s列的作答：%s" % choice_pos_n_ans
            else:
                print u"第%s排第%s列的作答：可能没有填涂" % choice_pos
