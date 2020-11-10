import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import glob

def getImage():
    images = [cv2.imread(file) for file in glob.glob(r'C:\Users\Kevin\Dropbox\UCTSHITS\4022S\Code\Kevin\ShapeDetection\data5\*.png')]
    imgname = []
    for file in glob.glob(r'C:\Users\Kevin\Dropbox\UCTSHITS\4022S\Code\Kevin\ShapeDetection\data5\*.png'):
        imgname.append(file)
    return images, imgname

def writeImage(img, imgName):
    pathWrite = r'C:\Users\Kevin\Dropbox\UCTSHITS\4022S\Code\Kevin\ShapeDetection\rSignData5\\'
    # pathWrite = pathWrite[:-1] + 'Output\\' 
    imagePath = pathWrite + "\\data5_" + imgName[-8:]
    cv2.imwrite(imagePath, img)

def convertRGB(img):
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgbimg

def FilterRed(img):
    # red colour threshold
    lowerRedRange = (105,40,25)
    higherRedRange = (170,60,40)
    maskRed = cv2.inRange(img, lowerRedRange, higherRedRange)

    lowerRedRange2 = (80,10,10)
    higherRedRange2 = (130,40,45)
    maskRed2 = cv2.inRange(img, lowerRedRange2, higherRedRange2)

    lowerRedRange3 = (190,80,45)
    higherRedRange3 = (205,90,80)
    maskRed3 = cv2.inRange(img, lowerRedRange3, higherRedRange3)

    com = cv2.bitwise_or(maskRed, maskRed2)
    com = cv2.bitwise_or(com, maskRed3)

    # plt.imshow(maskRed, cmap = 'gray')
    # plt.show()
    return com

# #RED FILTER DATA 4
# def FilterRed(img):
#     # red colour threshold
#     lowerRedRange = (80,50,58)
#     higherRedRange = (120,85,85)
#     maskRed = cv2.inRange(img, lowerRedRange, higherRedRange)

#     # lowerRedRange2 = (55,10,15)
#     # higherRedRange2 = (70,40,40)
#     # maskRed2 = cv2.inRange(img, lowerRedRange2, higherRedRange2)

#     # lowerRedRange2 = (35,8,10)
#     # higherRedRange2 = (45,18,25)
#     # maskRed2 = cv2.inRange(img, lowerRedRange2, higherRedRange2)

#     # com = cv2.bitwise_or(maskRed, maskRed2)
#     # plt.imshow(maskRed, cmap = 'gray')
#     # plt.show()
#     return maskRed

#BLUE FILTER DATA 4
# def FilterBlue(img):
#     lowerBlueRange = (60,110,185)
#     higherBlueRange = (110,160,240)
#     maskBlue = cv2.inRange(img, lowerBlueRange, higherBlueRange)

#     lowerBlueRange2 = (20,40,130)
#     higherBlueRange2 = (65,115,225)
#     maskBlue2 = cv2.inRange(img, lowerBlueRange2, higherBlueRange2)

#     lowerBlueRange3 = (5,25,90)
#     higherBlueRange3 = (15,50,110)
#     maskBlue3 = cv2.inRange(img, lowerBlueRange3, higherBlueRange3)
#     com = cv2.bitwise_or(maskBlue, maskBlue2)
#     com2 = cv2.bitwise_or(com, maskBlue3)
    

#     # plt.imshow(maskBlue2, cmap = 'gray')
#     # plt.show()
#     # return maskBlue2
#     return com2

def FilterBlue(img):
    lowerBlueRange = (30,60,125)
    higherBlueRange = (30,60,125)
    # lowerBlueRange2 = (30,40,150)
    # higherBlueRange2 = (60,100,210)
    maskBlue = cv2.inRange(img, lowerBlueRange, higherBlueRange)
    # maskBlue2 = cv2.inRange(img, lowerBlueRange2, higherBlueRange2)
    # com = cv2.bitwise_or(maskBlue, maskBlue2)
    # plt.imshow(maskBlue2, cmap = 'gray')
    # plt.show()
    # return maskBlue2
    return maskBlue

def FilterYellow(img):
    # lowerYellowRange = (45,38,15)
    # higherYellowRange = (75,55,35)
    lowerYellowRange = (75,55,35)
    higherYellowRange = (75,55,35)
    maskYellow = cv2.inRange(img, lowerYellowRange, higherYellowRange)
    # plt.imshow(maskBlue, cmap = 'gray')
    # plt.show()
    return maskYellow

def FilterWhite(img):
    lowerWhiteRange = (185,200,200)
    higherWhiteRange = (225,250,245)
    maskWhite = cv2.inRange(img, lowerWhiteRange, higherWhiteRange)

    lowerWhiteRange2 = (150,170,180)
    higherWhiteRange2 = (155,190,190)
    maskWhite2 = cv2.inRange(img, lowerWhiteRange2, higherWhiteRange2)

    lowerWhiteRange3 = (130,140,140)
    higherWhiteRange3 = (150,170,175)
    maskWhite3 = cv2.inRange(img, lowerWhiteRange3, higherWhiteRange3)

    lowerWhiteRange4 = (150,180,180)
    higherWhiteRange4 = (170,195,210)
    maskWhite4 = cv2.inRange(img, lowerWhiteRange4, higherWhiteRange4)

    lowerWhiteRange5 = (100,110,110)
    higherWhiteRange5 = (130,145,148)
    maskWhite5 = cv2.inRange(img, lowerWhiteRange5, higherWhiteRange5)

    lowerWhiteRange6 = (70,80,90)
    higherWhiteRange6 = (105,115,115)
    maskWhite6 = cv2.inRange(img, lowerWhiteRange6, higherWhiteRange6)

    com = cv2.bitwise_or(maskWhite, maskWhite2)
    com = cv2.bitwise_or(com, maskWhite3)
    com = cv2.bitwise_or(com, maskWhite4)
    com = cv2.bitwise_or(com, maskWhite5)
    com = cv2.bitwise_or(com, maskWhite6)
    # plt.imshow(com, cmap = 'gray')
    # plt.show()
    return com

def detectContour(filteredImg):
    blurred = cv2.GaussianBlur(filteredImg, (9, 11), 0)
    edge = cv2.Canny(blurred, 100, 300)
    
    kernel = np.ones((6,6), np.uint8) 
    kernel2 = np.ones((3,3), np.uint8) 
    dilation = cv2.dilate(edge, kernel, iterations = 3)
    erosion = cv2.erode(dilation, kernel2, iterations = 3)
    # plt.imshow(edge, cmap = 'gray')
    # plt.show()
    # plt.imshow(dilation, cmap = 'gray')
    # plt.show()
    contours, hierarchy = cv2.findContours(dilation,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours

def areaThresh(contours):
    max = 1000
    for c in contours:
        a = cv2.contourArea(c)
        if a > max:
            max = a
    thresh = max*(1/2)
    return thresh

    #     cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    #     cv2.imshow('Contours', img) 
    #     cv2.waitKey(0) 
    # cv2.destroyAllWindows()

def maskSign(img, contours):
    mask = np.zeros_like(img)
    mask2 = np.zeros_like(img)
    thresh = areaThresh(contours)
    for c in contours:
        a = cv2.contourArea(c)

        # if (((len(approx) >= 3) and (len(approx) < 5)) or (len(approx) > 10)):

            
        if a > thresh:
            print('thresh: ', thresh)
            print(a)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1*peri, True)
            print(len(approx))
            if len(approx) > 2:
                cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
                m = cv2.moments(c)
                # cX = int(m["m10"]/m["m00"])
                # cY = int(m["m01"]/m["m00"])
                # print(cX, ", ", cY)
                cv2.imshow("Image", mask)
                cv2.waitKey(0)
    plt.imshow(mask, cmap="gray")
    plt.show()
    
    mask = cv2.rectangle(mask, (0,260), (1200, 376), (0, 0, 0), -1)
    mask2 = cv2.rectangle(mask, (0,0), (670, 376), (0, 0, 0), -1)
    mask = cv2.bitwise_or(mask, mask2)

    # mask2 = cv2.rectangle(mask2, (150,0), (670, 150), (255, 255, 255), -1)
    # mask = cv2.bitwise_and(mask, mask2)

    mask = cv2.bitwise_not(mask)
    


    # plt.imshow(mask, cmap = 'gray')
    # plt.show()
    return mask

def removeSign(img, mask):
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]
    return out

def combineImg(img1, img2):
    combine = cv2.bitwise_and(img1, img2)
    return combine

def main():
    images, imgname = getImage()
    x=0
    # plt.imshow(images[0])
    # plt.show()
    for img in images:
        
        img = convertRGB(img)
        print(img.shape)
        # blueImg = FilterBlue(img)
        redImg = FilterRed(img)
        # yellowImg = FilterYellow(img)
        whiteImg = FilterWhite(img)

        # blueContours = detectContour(blueImg)
        redContours = detectContour(redImg)
        # yellowContours = detectContour(yellowImg)
        whiteContours = detectContour(whiteImg)

        # blueMask = maskSign(img, blueContours)
        redMask = maskSign(img, redContours)
        # YellowMask = maskSign(img, yellowContours)
        whiteMask = maskSign(img, whiteContours)

        # combineMask = combineImg(blueMask, redMask)
        # combineMask = combineImg(combineMask, YellowMask)
        combineMask = combineImg(whiteMask, redMask)

        # output = removeSign(img, whiteMask)
        output = removeSign(img, combineMask)
        output = convertRGB(output)
        print(x)
        print(imgname[x])
        writeImage(output, imgname[x])
        
        # plt.imshow(output)
        # plt.show()
        break
        x = x + 1




    

    # img = convertRGB(images[2])
    # img2 = convertRGB(images[0])
    
    # blueImg = FilterBlue(img2)
    # contours = detectContour(blueImg)
    # output = removeSign(images[0], contours)
    # # redImg = FilterRed(img)
    # # contours = detectContour(redImg)
    # # output = removeSign(images[2], contours)





if __name__ == '__main__':
    main()