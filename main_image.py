from engine import engine
import cv2
import numpy as np
import keyboard

#red color ranges
lower_red1 = np.array([0,100, 100])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([160,100, 25])
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
stop_path = "./Template/stop"
stop_engine = engine(stop_path, lower_red1, upper_red1, lower_red2, upper_red2, "stop",0.5)
engines.append(stop_engine)

#yeild engine
yeild_path = "./Template/yeild"
yeild_engine = engine(yeild_path, lower_red1, upper_red1, lower_red2, upper_red2, "yeild",1)
engines.append(yeild_engine)

#warning engine
warn_path = "./Template/warn"
warn_engine = engine(warn_path, lower_yellow1, upper_yellow1, lower_yellow2, upper_yellow2, "warning",1.5)
engines.append(warn_engine)

#direction engine
direction_path = "./Template/direction"
direction_engine = engine(direction_path, lower_green1, upper_green1, lower_green2, upper_green2, "direction",2)
engines.append(direction_engine)

#initilaize engines
for engine in engines:
    engine.create_templates()
    engine.create_templates_mask()

# compute image result
frame = cv2.imread("./data/direction_test.jpg")
img = frame
result = []
val = []
for engine in engines:
    result_temp,val_temp,temp = engine.compute(frame)
    if result_temp == True:
        img = cv2.bitwise_or(img, temp)
    result.append(result_temp)
    val.append(val_temp)

cv2.imshow('image', img)
print(result)
print(val)
cv2.waitKey(0)



