from xlutils.copy import copy
import xlrd as ExcelRead


def write_scores(file_name, choices, pic_name, gray):
    r_xls = ExcelRead.open_workbook(file_name)
    r_sheet = r_xls.sheet_by_index(0)
    rows = r_sheet.nrows

    no_sheet = r_xls.sheet_by_index(3)
    no_rows = no_sheet.nrows

    w_xls = copy(r_xls)
    sheet_write = w_xls.get_sheet(0)
    sheet_wt_ch = w_xls.get_sheet(2)
    sheet_rt_ans = r_xls.sheet_by_index(1)
    sheet_no = w_xls.get_sheet(3)
    if choices:
        sheet_write.write(rows, 0, pic_name)
        for i in range(1, len(choices) + 1):
            score = 0
            try:
                if choices[i - 1] == sheet_rt_ans.cell(i - 1, 0).value:
                    score = 1
            except IndexError:
                score = ''
            sheet_write.write(rows, i, score)
            # sheet_write.write(rows, i, choices[i - 1])
            sheet_wt_ch.write(rows, i, choices[i - 1])
        sheet_write.write(rows, i + 1, gray)
        w_xls.save(file_name)
    else:
        sheet_no.write(no_rows, 0, pic_name)
        w_xls.save(file_name)