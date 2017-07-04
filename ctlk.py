import pyautogui   
import cv2
import numpy as np
from matplotlib import pyplot as plt

s = pyautogui.screenshot()
s.save('screenshot1.png')
img_rgb = cv2.imread('screenshot1.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('sample.PNG',0)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_n = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

edges = cv2.Canny(img_n,50,150,apertureSize = 3)

oi = 0
temp = []

w0, h0 = template.shape[::-1]

threshold = 0.7

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
loc = np.where( res >= threshold)
temp = loc
minLineLength = 100
maxLineGap = 10
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#for x1,y1,x2,y2 in lines[0]:
#   cv2.line(img_rgb,(x1,y1),(x2,y2),(0,255,0),1)

for pt in zip(*temp[::-1]):
	   cv2.rectangle(img_rgb, pt, (pt[0] + w0, pt[1] + h0), (0,0,0), 2)



cv2.imwrite('res1.png',img_n)

cv2.imwrite('res.png',img_rgb)


