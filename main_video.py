from engine import engine
import cv2
import numpy as np

#red color ranges
lower_red1 = np.array([0,100, 100])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([160,100, 100])
upper_red2 = np.array([179,255,255])

#yellow color ranges
lower_yellow1 = np.array([16,200, 100])
upper_yellow1 = np.array([28,255,255])
lower_yellow2 = np.array([10,200, 100])
upper_yellow2 = np.array([28,255,255])

#green color rannges
lower_green1 = np.array([70,100,50])
upper_green1 = np.array([90,255,255])
lower_green2 = np.array([80,100, 50])
upper_green2 = np.array([90,255,255])

#engines list
engines = []

#stop engine
stop_path = ".\emplate\stop_temp"
stop_engine = engine(stop_path, lower_red1, upper_red1, lower_red2, upper_red2, "stop",0.5)
engines.append(stop_engine)

#yeild engine
yeild_path = ".\emplate\yeild_temp"
yeild_engine = engine(yeild_path, lower_red1, upper_red1, lower_red2, upper_red2, "yeild",1)
engines.append(yeild_engine)

#warning engine
warn_path = ".\emplate\warn_temp"
warn_engine = engine(warn_path, lower_yellow1, upper_yellow1, lower_yellow2, upper_yellow2, "warning",1.5)
engines.append(warn_engine)

#direction engine
direction_path = ".\emplate\direc_temp"
direction_engine = engine(direction_path, lower_green1, upper_green1, lower_green2, upper_green2, "direction",2)
engines.append(direction_engine)

#initilaize engines
for engine in engines:
    engine.create_templates()
    engine.create_templates_mask()

# compute image result
cap = cv2.VideoCapture('.\data\demo_video_fail.avi')

while (cap.isOpened()) :
    _, frame = cap.read()
    img = frame
    for engine in engines:
        result_temp,val_temp,temp = engine.compute(frame)
        if result_temp == True:
            img = cv2.bitwise_or(img, temp)
    cv2.imshow('frame', img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

