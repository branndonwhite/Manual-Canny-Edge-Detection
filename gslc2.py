import cv2
import numpy as np
import matplotlib.pyplot as plt

def showImg(row, col, img,label):
    plt.figure(1, figsize=(8,8))
    for i,(curr_img,curr_label) in enumerate(zip(img,label)):
        plt.subplot(row,col,(i+1))
        plt.imshow(curr_img, 'gray')
        plt.title(curr_label)
        plt.xticks([])
        plt.yticks([])
    plt.show()

img = cv2.imread('lena.jpg')
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Noise Reduction
#denoise = cv2.fastNlMeansDenoising(im, None, 10, 7, 21)
blur = cv2.GaussianBlur(im, (5,5), 1.4)

#Gradient Calculation
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

gradient = np.hypot(sobelx, sobely)
gradient = (gradient / gradient.max() * 255) - 0 
theta = np.arctan2(sobelx, sobely)

#Non-Maximum Suppression
nms = np.zeros(gradient.shape, dtype=np.int32)
angle = theta * 180. / np.pi
angle[angle < 0] += 180

for i in range(1, int(gradient.shape[0] - 1)):
    for j in range(1, int(gradient.shape[1] - 1)):
        #angle 0
        if (0 <= angle[i,j] <= 22.5) or (157.5 <= angle[i,j] <= 180):
                if (gradient[i,j] > gradient[i+1,j]) and (gradient[i,j] > gradient[i-1,j]):
                        nms[i,j] = gradient[i,j] 
                else:
                        nms[i,j] = 0
        #angle 45
        elif (22.5 <= angle[i,j] < 67.5):
                if (gradient[i,j] > gradient[i+1,j+1]) and (gradient[i,j] > gradient[i-1,j-1]):
                        nms[i,j] = gradient[i,j]
                else:
                        nms[i,j] = 0
        #angle 90
        elif (67.5 <= angle[i,j] < 112.5):
                if (gradient[i,j] > gradient[i,j+1]) and (gradient[i,j] > gradient[i,j-1]):
                        nms[i,j] = gradient[i,j]
                else:
                        nms[i,j] = 0
        #angle 135
        elif (112.5 <= angle[i,j] < 157.5):
                if (gradient[i,j] > gradient[i+1,j-1]) and (gradient[i,j] > gradient[i-1,j+1]):
                        nms[i,j] = gradient[i,j]
                else:
                        nms[i,j] = 0
                        
#Double Threshold
lowThresRatio = 0.01
highThresRatio = 0.1

highThreshold = nms.max() * highThresRatio
lowThreshold = highThreshold * lowThresRatio

thres = np.zeros(nms.shape, dtype=np.int32)

weak = np.int32(100)
strong = np.int32(255)

strongi, strongj = np.where(nms >= highThreshold)
zeroi, zeroj = np.where(nms < lowThreshold)
weaki, weakj = np.where((nms >= lowThreshold) & (nms <= highThreshold))

thres[strongi, strongj] = strong
thres[weaki, weakj] = weak

#Hysteresis Thresholding
strong = 255
hyst = thres.copy()

for i in range(1, int(thres.shape[0] - 1)):
        for j in range(1, int(thres.shape[1] - 1)):
                if (thres[i,j] == weak):
                        if ((thres[i,j+1] == strong) or (thres[i,j-1] == strong)
                                or (thres[i+1,j] == strong) or (thres[i-1,j] == strong)
                                or (thres[i+1,j-1] == strong) or (thres[i-1,j+1] == strong)
                                or (thres[i+1,j+1] == strong) or (thres[i-1,j-1] == strong)):
                                hyst[i,j] = strong
                        else:
                                hyst[i,j] = 0 

#Built-in Canny
canny = cv2.Canny(im, 100, 200)

img_blur = [im, blur] 
label_blur = ['Grayscale', 'Denoising']
#showImg(1,2,img_blur, label_blur)

img_gradient = [sobelx, sobely, gradient]
label_gradient = ['Sobel X', 'Sobel Y', 'Gradient'] 
#showImg(1,3,img_gradient,label_gradient)

img_nms = [nms, thres]
label_nms = ['Non-Maxium Suppression', 'Double Thresholding']
#showImg(1,2,img_nms,label_nms)

img_canny = [hyst, canny]
label_canny = ['Hysteresis','Built-In Canny']
showImg(1,2,img_canny,label_canny)

#source:
#https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
#https://github.com/MadhavEsDios/Canny-Edge-Detector/blob/master/Canny%20Edge%20Detector.ipynb
#http://justin-liang.com/tutorials/canny/