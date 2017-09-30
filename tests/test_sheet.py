# coding=utf-8
from __future__ import unicode_literals
from sheet import get_answer_from_sheet
import cv2
import os
from write import write_scores

for dir_path, sub_paths, files in os.walk('imgs'):
    for f in files:
        file_path = os.path.join(dir_path, f)
        print file_path
        base_img = cv2.imread(file_path)
        try:
            scores, gray = get_answer_from_sheet(base_img)
        except Exception as e:
            print e.message
            scores, gray = None, None
        # scores, gray = get_answer_from_sheet(base_img)
        _path = os.path.split(dir_path)[1]
        write_scores('score/' + _path + '.xls', scores, file_path, gray)
