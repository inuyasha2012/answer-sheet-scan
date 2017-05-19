# coding=utf-8
import numpy as np

# 选项
CHOICES = "ABCDE"

# 一行选项+题号列数，例如一行有3题，一题4个选项，所以总共有3*4+3个列
CHOICE_COL_COUNT = 18

# 每题题选项数
CHOICES_PER_QUE = 5

# 每个选项框里面白色点所占比例阈值，小于则说明该选项框可能被填涂
WHITE_RATIO_PER_CHOICE = 0.80

# 受限于环境，光源较差的情况下或腐蚀膨胀参数设置不对，
# 可能会有误判，这个参数这是比较两个都被识别为涂写的选项框是否有误判的阈值
MAYBE_MULTI_CHOICE_THRESHOLD = 0.07

# 答题卡框与整个图片周长比的阈值
CNT_PERIMETER_THRESHOLD = 0.35

# 答题卡框面积阈值
SHEET_AREA_MIN_RATIO = 0.7

# 识别所涂写区域时的二值化参数
ANS_IMG_THRESHOLD = (88, 255)

# 识别所涂写区域时的膨胀参数
ANS_IMG_DILATE_ITERATIONS = 9

# 识别所涂写区域时的腐蚀参数
ANS_IMG_ERODE_ITERATIONS = 0

# 识别所涂写区域时的膨胀腐蚀的kernel
ANS_IMG_KERNEL = np.ones((2, 2), np.uint8)

# 识别所有选项框区域时的二值化参数
CHOICE_IMG_THRESHOLD = (115, 255)

# 识别所有选项框区域时的膨胀参数
CHOICE_IMG_DILATE_ITERATIONS = 6

# 识别所有选项框区域时的腐蚀参数
CHOICE_IMG_ERODE_ITERATIONS = 3

# 识别所有选项框区域时的膨胀腐蚀的kernel
CHOICE_IMG_KERNEL = np.ones((2, 2), np.uint8)

# 选项框面积的阈值，超过则认为这个轮廓不是选项框
CHOICE_MAX_AREA = 400

# 选项框面积的阈值，小于则认为这个轮廓不是选项框
CHOICE_MIN_AREA = 100

# 总共选项框 + 题号的个数，例如一行3题，总共20列，所以有3 * 20 * 4 + 3 * 20
CHOICE_CNT_COUNT = 51* 6

# 调整亮度的竖向分块数目
PROCESS_BRIGHT_COLS = 18

# 调整亮度的横向分块数目
PROCESS_BRIGHT_ROWS = 16

# 调整亮度值
BRIGHT_VALUE = 120

test_ans = ['A', 'AD', 'BD', 'AC', 'B', 'BD', 'A', '', 'AD', 'D', '', '', 'B', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A',
            'B', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', '', '']

ORIENT_CODE = {'col': 1, 'row': 0}