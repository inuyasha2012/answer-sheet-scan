import cv2
from sheet import get_answer_from_sheet

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    cv2.imshow("img", frame)
    height, width = frame.shape[:2]
    frame = cv2.resize(
        frame,
        # (int(round(0.7667 * width)), int(round(0.765625 * height))),*
        (600, 500),
        interpolation=cv2.INTER_CUBIC
    )
    try:
        get_answer_from_sheet(frame)
        break
    except Exception as e:
        # print e
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue


cap.release()
cv2.destroyAllWindows()