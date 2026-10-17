import cv2
import numpy as np
import mediapipe as mp

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)
canvas = None
prev = None
smooth = None
color = (0, 0, 255)  # BGR

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if canvas is None:
        canvas = np.zeros_like(frame)

    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        raw = np.array([lm[8].x * w, lm[8].y * h])

        # smoothing so the line isn't shaky
        smooth = raw if smooth is None else smooth * 0.6 + raw * 0.4
        tip = (int(smooth[0]), int(smooth[1]))

        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y

        if index_up and not middle_up:       # draw mode
            if prev:
                cv2.line(canvas, prev, tip, color, 6)
            prev = tip
            cv2.circle(frame, tip, 10, color, -1)
        else:                                 # pen up
            prev = None
            cv2.circle(frame, tip, 10, (255, 255, 255), 2)
    else:
        prev = None
        smooth = None

    # overlay canvas on frame
    mask = canvas.any(axis=2)
    frame[mask] = canvas[mask]

    cv2.imshow("Air Draw", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k == ord("c"):
        canvas[:] = 0
    elif k == ord("1"):
        color = (0, 0, 255)
    elif k == ord("2"):
        color = (0, 255, 0)
    elif k == ord("3"):
        color = (255, 0, 0)
    elif k == ord("e"):
        color = (0, 0, 0)  # eraser

cap.release()
cv2.destroyAllWindows()