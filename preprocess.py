# Patrick Chao
# 12/31/17
# Image renaming and removing black and white images
import os
from os import listdir,makedirs
from os.path import isfile,join
from PIL import Image
import random
import cv2
import numpy as np

def rename(filepath):
    blackWhiteFolder="BlackWhite"
    index = 20214
    bwIndex=522
    for folder in os.listdir(filepath):
        if folder!=".DS_Store":
            for filename in os.listdir(filepath +"/"+folder):
                if filename!=".DS_Store":
                    totalPath=filepath+"/"+folder+"/"+filename
                    img = Image.open(totalPath)
                    if isColor(img):
                        os.rename(totalPath,filepath+"/"+str(index)+".jpg")
                        print("color")
                        index += 1
                    else:
                        os.rename(totalPath,blackWhiteFolder+"/"+str(bwIndex)+".jpg")
                        print("bw")
                        bwIndex += 1
def val(filepath,dstpath,numImages=3000):
    for i in range(1,numImages+1):
        currImage=random.choice(os.listdir(filepath))
        newPath = join(dstpath,str(i)+".jpg")
        os.rename(os.path.join(filepath,currImage),newPath)
#Determines whether an image is black white
def isColor(img):
    w, h = img.size
    nrand=50
    x=np.random.choice(w,nrand).tolist()
    y=np.random.choice(h,nrand).tolist()
    coords=zip(x,y)
    for coord in coords:
        #is color
        if checkRGBEquality(img,coord[0],coord[1]):
            return True
    #Is black white
    return False

#Determines if the pixel at x,y of image is grayscale
#Check if they all are within 6 of each other
def checkRGBEquality(img,x,y):
    r,g,b=img.getpixel((x,y))
    #Black white
    if abs(r-g)<=6 and abs(r-b)<=6 and abs(g-b)<=6:
        return False

    #Color
    return True

#change names from 1 to 000001
def changeNames(filepath):
    for filename in os.listdir(filepath):
        if filename!=".DS_Store":
            totalPath=filepath+"/"+filename

            newName="0"*max((6-len(filename[:-4])),0)+filename
            newPath=filepath+"/"+newName
            os.rename(totalPath,newPath)


            import cv2

#https://stackoverflow.com/questions/47087528/converting-images-in-a-folder-to-grayscale-using-python-and-opencv-and-writing-i
def makeGrayscale(path,dstpath):
    files = [f for f in listdir(path) if isfile(join(path,f))]
    i = 0
    for image in files:

        if image!=".DS_Store":
            if i<1:
                i=1

                img = cv2.imread(os.path.join(path,image))
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                dstPath = join(dstpath,image)
                cv2.imwrite(dstPath,gray)

if __name__ == "__main__":
    # rename("reformatted")
    # changeNames("val/valImages")
    makeGrayscale("train/train","grayscale/train")
    #val("train/train","val/valImages")
