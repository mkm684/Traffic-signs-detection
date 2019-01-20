import cv2
import numpy as np
class engine:
    def __init__(self, file_path, range1_low, range1_up, range2_low, range2_up, type, num):
        self.file_path = file_path
        self.range1_low = range1_low
        self.range1_up = range1_up
        self.range2_low = range2_low
        self.range2_up = range2_up
        self.type = type
        self.templates, self.templates_mask, self.templates_hsv = ([] for i in range(3))
        self.kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        self.kernel = np.ones((15,15), np.uint8)
        self.num = num
        if self.type == "direction":
            self.threshld = 1
        else:
            self.threshld = 0.1

    def create_templates(self):
        for i in range(0,4):
            path = self.file_path+str(i+1)+".jpg"
            stop_template = cv2.imread(path)
            self.templates.append(stop_template.copy())
            self.templates_hsv.append(cv2.cvtColor(stop_template, cv2.COLOR_BGR2HSV))

    def create_templates_mask(self):
        for i in range(0,4):
            mask1 = cv2.inRange(self.templates_hsv[i], self.range1_low, self.range1_up)
            mask2 = cv2.inRange(self.templates_hsv[i], self.range2_low, self.range2_up)
            self.templates_mask.append(cv2.bitwise_or(mask1,mask2))

    def filter(self, mask):
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_cross)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
        return closing

    def max_contours(self, filtered_img):
        im1, contours, hierarchy = cv2.findContours(filtered_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0
        c = max(contours, key = cv2.contourArea)
        if cv2.contourArea(c) <= 500:
            return 0
        return c

    def compare(self, c):
        sum = 0
        for a in range(0,4):
            b = self.max_contours(self.filter(self.templates_mask[a]))
            cv2.drawContours(self.templates[a], b, -1, (0,255,0), 3)
            sum += cv2.matchShapes(b, c, 1, 0.0)
        if sum/4 <= self.threshld:
            return True, sum/4
        else:
            return False, sum/4

    def compute(self, frame):
        im2copy = frame.copy()
        height, _, _ = im2copy.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.range1_low, self.range1_up)
        mask2 = cv2.inRange(hsv, self.range2_low, self.range2_up)
        mask = cv2.bitwise_or(mask1,mask2)
        cv2.imshow("segmentation", mask)
        filtered_image = self.filter(mask)
        cv2.imshow("morphogical", filtered_image)
        image_contours = self.max_contours(filtered_image)
        if image_contours is 0:
            return False, 100, im2copy
        cv2.drawContours(im2copy, image_contours, -1, (255,255,255), 3)
        cv2.putText(im2copy,self.type,(10,int(self.num*(height/4))), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        result_temp,val_temp = self.compare(image_contours)
        return result_temp,val_temp,im2copy


