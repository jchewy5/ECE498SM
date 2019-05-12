import numpy as np
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import math

testImageRoot = "test_images/sidewalk5.JPG"

rawImg = cv2.imread(testImageRoot)
rawImg = cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB)
rawImage = cv2.imread(testImageRoot)

x1, y1 = 93, 2643
x2, y2 = 846, 418
x3, y3 = 2896, 2643
x4, y4 = 2200, 418

cv2.circle(rawImage,(x1,y1), 40, (0,0,255), -1)
cv2.circle(rawImage,(x2,y2), 40, (0,0,255), -1)
cv2.circle(rawImage,(x3,y3), 40, (0,0,255), -1)
cv2.circle(rawImage,(x4,y4), 40, (0,0,255), -1)

plot1 = cv2.cvtColor(rawImage,cv2.COLOR_BGR2RGB)


def warpImage(img):  
    h,w = img.shape[:2]
    x1, y1 = 93, 2643
    x2, y2 = 846, 418
    x3, y3 = 2896, 2643
    x4, y4 = 2200, 418

    dst = np.float32([(500,0),
                  (w-500,0),
                  (500,h),
                  (w-500,h)])   
    src = np.float32([(x2,y2),(x4,y4),(x1,y1),(x3,y3)])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warpedImg = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)

    return warpedImg, M, Minv


def sobel(img, thresh_min=0, thresh_max=255):
    sobelX = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    scaled_lap = np.uint8(255*sobelX/np.max(sobelX))
    binary_output = np.zeros_like(scaled_lap)
    binary_output[(scaled_lap >= thresh_min) & (scaled_lap <= thresh_max)] = 1

    return binary_output   


def combinedEdgeDetection(img):
    blur = cv2.bilateralFilter(img, 10, 200, 200)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

    kernel = np.ones((9,9),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    v_kernel = np.ones((20, 1),np.uint8)
    h_kernel = np.ones((1, 20),np.uint8)

    v_dilation = cv2.dilate(bw, v_kernel, iterations=3)
    erosion = cv2.erode(v_dilation, h_kernel, iterations=5)
    v_dilation2 = cv2.dilate(erosion, v_kernel, iterations=2)
    erosion2 = cv2.erode(v_dilation2, h_kernel, iterations=2)
    dilation2 = cv2.dilate(erosion2, v_kernel, iterations=2)

    sobel_x = np.uint8(np.abs(cv2.Sobel(dilation2, cv2.CV_64F, 1, 0, ksize=5)))
    sobel_y = np.uint8(np.abs(cv2.Sobel(dilation2, cv2.CV_64F, 0, 1, ksize=5)))
    sobel = cv2.bitwise_or(sobel_x,sobel_y)
    plt.imshow(sobel)
    plt.show()
    sobel_dilation = cv2.dilate(sobel, v_kernel, iterations=2)
    sobel_dilation2 = cv2.erode(sobel_dilation, h_kernel, iterations=1)

    return sobel_dilation2

window_width = 30 
window_height = rawImg.shape[0] / 10.0
margin = 40
window = np.ones(window_width)
convThres = 100
verticle_ratio = 1.0/2 
horizontal_ratio = 0.15

def findConvCenter(image):
    leftCentroids = []
    rightCentroids = []

    vslice = np.sum(image[(int(image.shape[0]/2)):,:],axis=0)
    newslice = np.convolve(vslice,window)
    leftCenterX = (np.argmax(np.split(newslice,[int(len(newslice)/2)])[0]))
    rightCenterX = int(len(newslice)/2) + (np.argmax(np.split(newslice,[int(len(newslice)/2)])[1]))
    maxleft = newslice[leftCenterX]/5
    maxright = newslice[rightCenterX]/5
    leftCenterY = np.argmax(image[:,leftCenterX])
    rightCenterY = np.argmax(image[:,rightCenterX])

    for i in range(5):
        leftCentroids.append((leftCenterX, leftCenterY))
        rightCentroids.append((rightCenterX, rightCenterY))

    leftEmptyCount, rightEmptyCount = 0, 0
    for level in range(1,(int)(image.shape[0]/window_height)):
        vslice = np.sum(image[int((level-1)*window_height):int(level*window_height),:],axis=0)
        c = np.convolve(vslice,window)
        oldleftCenterX = leftCenterX
        eligibleRange = c[oldleftCenterX-margin:oldleftCenterX+margin]
        if eligibleRange.size:
            leftCenterX = oldleftCenterX -margin + np.argmax(eligibleRange)
            leftCenterY = int((level-1)*window_height + (window_height/2))

            oldrightCenterX = rightCenterX
            eligibleRange = c[oldrightCenterX-margin:oldrightCenterX+margin]
            rightCenterX = oldrightCenterX -margin + np.argmax(eligibleRange)
            rightCenterY = int((level-1)*window_height + (window_height/2))

            leftCentroids.append((leftCenterX, leftCenterY))
            rightCentroids.append((rightCenterX, rightCenterY))
        else:
            continue

    return leftCentroids, rightCentroids

def window_mask(width, height, img_ref, centerX, centerY):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-centerY-height/2):int(img_ref.shape[0]-centerY+height/2),max(0,int(centerX-width/2)):min(int(centerX+width/2),img_ref.shape[1])] = 1
    return output

def plotWindows(warped):
    leftCentroids, rightCentroids= findConvCenter(warped)
    r_points = np.zeros_like(warped)
    l_points = np.zeros_like(warped)    
    
    if len(leftCentroids) > 0 and len(rightCentroids) > 0:
        if len(leftCentroids) > 0:
            for level in range(0,len(leftCentroids)):
                l_mask = window_mask(window_width,window_height,warped,leftCentroids[level][0],leftCentroids[level][1])
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255

        if len(rightCentroids) > 0:
            for level in range(0,len(rightCentroids)):
                r_mask = window_mask(window_width,window_height,warped,rightCentroids[level][0],rightCentroids[level][1])
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        template = np.array(r_points+l_points,np.uint8)
        zero_channel = np.zeros_like(template)
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
        warpage= 255*(np.dstack((warped, warped, warped))*255)
        output = cv2.addWeighted(warpage, 0.5, template, 0.5, 0.0)

    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    
    return output


def drawPoly(originImg, warpImg, leftCentroids, rightCentroids, Minv):
    leftCentroids = np.array(leftCentroids)
    rightCentroids = np.array(rightCentroids)
    leftpoly = np.polyfit(leftCentroids[:,0],leftCentroids[:,1],1)
    rightpoly = np.polyfit(rightCentroids[:,0],rightCentroids[:,1],1)

    plotY = np.linspace(0, warpImg.shape[0]-1, warpImg.shape[0])
    leftPlotX = np.array([np.roots((leftpoly-[0,y]))[0] for y in plotY])
    rightPlotX = [np.roots((rightpoly-[0,y]))[0] for y in plotY]

    leftPlotX[leftPlotX<=0] = -1
    leftPlotX[leftPlotX>=len(originImg[0])] = -1
    delidx = np.where(leftPlotX==-1)
    leftPlotX = np.delete(leftPlotX,delidx)
    plotY = np.delete(plotY,delidx)
    rightPlotX = np.delete(rightPlotX,delidx)
    rightPlotX[rightPlotX<=0] = -1
    rightPlotX[rightPlotX>=len(originImg[0])] = -1
    delidx = np.where(rightPlotX==-1)
    leftPlotX = np.delete(leftPlotX,delidx)
    plotY = np.delete(plotY,delidx)
    rightPlotX = np.delete(rightPlotX,delidx)

    leftPts = np.array([np.transpose(np.vstack([leftPlotX, plotY]))],dtype='int32')
    rightPts = np.array([np.flipud(np.transpose(np.vstack([rightPlotX, plotY])))],dtype='int32')
    points = np.hstack((leftPts, rightPts))

    bg = np.zeros_like(originImg)
    newImg = cv2.fillPoly(bg,points,(255,0,0))

    newImg = cv2.warpPerspective(newImg, Minv, (len(newImg[0]),len(newImg)), flags=cv2.INTER_LINEAR)
    newImg = cv2.addWeighted(newImg, 0.3, originImg, 0.7, 0)

    return newImg, leftPlotX, rightPlotX

images = glob.glob("final_sidewalk/2019-04-30 19-39-24.jpg")
for frame in images:
    img = cv2.imread(frame)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    warpedImg, M, Minv = warpImage(img)
    edgeDetectedImg = combinedEdgeDetection(warpedImg)
    leftCentroids, rightCentroids = findConvCenter(edgeDetectedImg)
    plotedLinesImg, leftX, rightX = drawPoly(img, warpedImg, leftCentroids, rightCentroids, Minv)
    
    f = plt.figure(figsize= (20,15))
    f.add_subplot(1, 2, 1)
    plt.imshow(img)
    f.add_subplot(1, 2, 2)
    plt.imshow(plotedLinesImg)
    plt.show()

def imagePipeline(img):
    img = cv2.resize(img, (1080, 720))
    warpedImg, M, Minv = warpImage(undistortedImg)
    edgeDetectedImg = combinedEdgeDetection(warpedImg)
    leftCentroids, rightCentroids = findConvCenter(edgeDetectedImg)
    imgOut, leftX, rightX = drawPoly(undistortedImg, warpedImg, leftCentroids, rightCentroids, Minv)
    return imgOut
    
