# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:58:56 2020

@author: s4100
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore
from myui import Ui_MainWindow
from  PyQt5.QtCore import pyqtSlot
from PIL import Image,ImageFilter
import cv2
import numpy as np
import os

class Controller(QMainWindow, Ui_MainWindow):
    def __init__(self, parent = None):
        super(QMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton11.clicked.connect(self.pushButton11_clicked)
        self.pushButton12.clicked.connect(self.pushButton12_clicked)
        self.pushButton21.clicked.connect(self.pushButton21_clicked)
        self.pushButton22.clicked.connect(self.pushButton22_clicked)
        self.pushButton23.clicked.connect(self.pushButton23_clicked)
        #self.comboBox.currentTextChanged.connect(self.combobox.currentData)
        #self.combo.activated.connect(self.handleActivated)
        self.pushButton24.clicked.connect(self.pushButton24_clicked)
        self.pushButton31.clicked.connect(self.pushButton31_clicked)
        self.pushButton31_2.clicked.connect(self.pushButton31_2_clicked)
        
        
        
        
    #第1-1
    def pushButton11_clicked(self):
        #讀入照片並灰階
        img=cv2.imread('./Hw2Dataset/Datasets/Q1_Image/coin01.jpg')
        img2=cv2.imread('./Hw2Dataset/Datasets/Q1_Image/coin02.jpg')
        
        #Gaussian模糊
        blur = cv2.GaussianBlur(img, (11,11), 0)
        blur2 = cv2.GaussianBlur(img2, (11,11), 0)
        
        #canny edge 取外框
        edged = cv2.Canny(blur,30,200)
        edged2 = cv2.Canny(blur2,30,200)
        
        #cv2.RETR_LIST 只取外層
        #cv2.CHAIN_APPROX_SIMPLE  只取長寬的end point
        (_, cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        (_, cnts2, _) = cv2.findContours(edged2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        #print("Number of Contours found = " + str(len(cnts)))
        #print("Number of Contours found = " + str(len(cnts2)))
        
        drawcontours = cv2.drawContours(img,cnts,-1,(0,255,255),5)
        drawcontours2 = cv2.drawContours(img2,cnts2,-1,(0,255,255),5)
        
        cv2.imshow('picture1',drawcontours)
        cv2.imshow('picture2',drawcontours2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    #第1-2   
    def pushButton12_clicked(self):
        
        img=cv2.imread('./Hw2Dataset/Datasets/Q1_Image/coin01.jpg')
        img2=cv2.imread('./Hw2Dataset/Datasets/Q1_Image/coin02.jpg')
        
        #Gaussian模糊
        blur = cv2.GaussianBlur(img, (11,11), 0)
        blur2 = cv2.GaussianBlur(img2, (11,11), 0)
        
        #canny edge 取外框
        edged = cv2.Canny(blur,30,200)
        edged2 = cv2.Canny(blur2,30,200)
        
        #cv2.RETR_LIST 只取外層
        #cv2.CHAIN_APPROX_SIMPLE  只取長寬的end point
        (_, cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        (_, cnts2, _) = cv2.findContours(edged2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        print("Number of Contours found = " + str(len(cnts)))
        print("Number of Contours found = " + str(len(cnts2)))
        
        
        #計算cnts
        str1 = str(len(cnts))
        str2 = str(len(cnts2))
        self.textEdit.append(str1)
        self.textEdit2.append(str2)

    
    
   
        
    #第2-1題
    def pushButton21_clicked(self):
        for i in range(len(arrayImg)):
            cv2.imshow(str(i+1),arrayImg[i])
            cv2.imwrite(str(i+1)+'.jpg',arrayImg[i])
    
    
    #第2-2題
    def pushButton22_clicked(self):
        print(intrinsic)
        
        
        
    #第2-3題
    def pushButton23_clicked(self):
       
        role = self.comboBox.currentIndex()
        A=int(role)
        print(type(A))
        print(A)
        print(extrinsic[A])
        #print(extrinsic2[A]) 
        
    #第2-4題
    def pushButton24_clicked(self):
        print(distortion)
        
    #第3-1題
    def pushButton31_clicked(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)#
        intrinsic=[]
        distortion=[]
        arrayImg=[]
        imgPath='./Hw2Dataset/Datasets/Q3_Image/'
        for imgName in os.listdir(imgPath):
            img=cv2.imread(imgPath+imgName)
            arrayImg.append(img)
            
        X=12# Number of chessboard squares along the x-axis
        Y=9# Number of chessboard squares along the y-axis
        iX=X-1# Number of interior corners along x-axis
        iY=Y-1# Number of interior corners along y-axis
        
        objp = np.zeros((iX*iY,3), np.float32)#
        objp[:,:2] = np.mgrid[0:iX,0:iY].T.reshape(-1,2)#
        
        axis2 = np.float32([[1,1,0], [3,5,0], [5,1,0],[3,3,-3]]).reshape(-1,3)#四面體
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)#坐標軸
        
        def draw(img, corners, imgpts):#坐標軸
            corner = tuple(corners[0].ravel())
            img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
            img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
            img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
            return img
        
        def draw2(img, corners, imgpts):#四面體
            for i in range(4):
                for j in range(4):
                    if i != j :
                        img = cv2.line(img, tuple(imgpts[i].ravel()), tuple(imgpts[j].ravel()),(0,0,255),3)
            return img
        
        objpoints = []#
        imgpoints = []#
        for i in range(len(arrayImg)):
            gray = cv2.cvtColor(arrayImg[i], cv2.COLOR_BGR2GRAY)
            success, corners = cv2.findChessboardCorners(gray, (iX, iY), None)
            if success == True:
                objpoints.append(objp)#
                imgpoints.append(corners)#
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                #cv2.drawChessboardCorners(arrayImg[i], (iX,iY), corners2,success)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)#未校正
                
                result, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, mtx, dist)
                
                imgpts, jac = cv2.projectPoints(axis2, rvecs, tvecs, mtx, dist)
                
                draw2(arrayImg[i],corners2,imgpts)
        for i in range(len(arrayImg)):
            cv2.imshow('1',arrayImg[i])
            cv2.waitKey(500)
            cv2.imwrite('3_'+str(i+1)+'.jpg',arrayImg[i])

        
        
        
        
    #第四題    
    def pushButton31_2_clicked(self):
        #Baseline=178 mm兩相機之間的距離
        #focal length=2826 pixels 相機焦距
        #最大視差 int16
        #blocksize 要是積數
    
       
        def Depth(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                depth=(1-(disparity[y][x]/255))*Z
                text='Disparity:'+str(disparity[y][x])+'pixels Depth:'+str(depth)+'mm'
                cv2.rectangle(disparity,(0,0),(850,40),(255,255,255),-1)
                cv2.putText(disparity, text, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
     
        while(1):
            cv2.imshow('disparity', disparity)
            cv2.setMouseCallback('disparity',Depth)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    


       
        
               
        
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = Controller()
    MainWindow.show()
    
    
    intrinsic=[]
    distortion=[]
    extrinsic=[]
    extrinsic2=[]
    arrayImg=[]
    imgPath='./Hw2Dataset/Datasets/Q2_Image/'
    for imgName in os.listdir(imgPath):
        img=cv2.imread(imgPath+imgName)
        arrayImg.append(img)
        
    X=12# Number of chessboard squares along the x-axis
    Y=9# Number of chessboard squares along the y-axis
    iX=X-1# Number of interior corners along x-axis
    iY=Y-1# Number of interior corners along y-axis
    
    objp = np.zeros((iX*iY,3), np.float32)#
    objp[:,:2] = np.mgrid[0:iX,0:iY].T.reshape(-1,2)#
    
    for i in range(len(arrayImg)):
        objpoints = []#
        imgpoints = []#
        gray = cv2.cvtColor(arrayImg[i], cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(gray, (iX, iY), None)
        if success == True:
            objpoints.append(objp)#
            imgpoints.append(corners)#
            # Draw the corners
            cv2.drawChessboardCorners(arrayImg[i], (iX, iY), corners, success)
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        intrinsic.append(mtx)
        distortion.append(dist)
        
        corners3=np.ones((3,1))
        objp3=np.ones((4,1))
        
        C=cv2.Rodrigues(rvecs[0])
        B2=np.c_[C[0],tvecs[0]]
        extrinsic2.append(B2)
        
        for j in range(len(objpoints)):
            for i in range(2): 
                corners3[i][0]=corners[j][0][i]
            for k in range(3):
                objp3[k][0]=objp[j][k]
            
            A=np.dot(np.linalg.inv(mtx),corners3)
            #print(A.T[2])
            #print(np.linalg.pinv(objp3))
            B=np.dot(A,np.linalg.pinv(objp3))
            extrinsic.append(B)#未完成
            
            
    
    #第四題
    imgL = cv2.imread('./Hw2Dataset/Datasets/Q4_Image/imgL.png',0)
    imgR = cv2.imread('./Hw2Dataset/Datasets/Q4_Image/imgR.png',0)
    
    stereo = cv2.StereoBM_create(numDisparities=192, blockSize=17)
    disparity = stereo.compute(imgL,imgR)
    #print( disparity)
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity = cv2.resize(disparity,(940,640))
    

    F=2826
    B=178
    D=123
    
    Z=(F*B)/D
    
    
    sys.exit(app.exec_()) 
    
    