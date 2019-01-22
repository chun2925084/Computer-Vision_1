# -*- coding: utf-8 -*-

import sys
from project_ui import Ui_MainWindow
import cv2 as cv
import numpy as np
import glob
import os
from PyQt5.QtWidgets import QMainWindow, QApplication


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)

    # button for problem 1.1
    def on_btn1_1_click(self):
        # add your code here
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images= glob.glob('./images/CameraCalibration/*.bmp')
        i=1
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (8,11),None)
            # If found, add object points, image points (after refining them)
            i=i+1
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv.drawChessboardCorners(img, (8,11), corners2,ret)
                cv.namedWindow(chr(i),cv.WINDOW_GUI_NORMAL )
                cv.imshow(chr(i),img)
                cv.waitKey(500)

    cv.destroyAllWindows()




    def on_btn1_2_click(self):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('./images/CameraCalibration/*.bmp')
        i=1
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv.drawChessboardCorners(img, (11,8), corners2,ret)
                # cv.namedWindow('img',cv.WINDOW_GUI_NORMAL )
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        for entry in mtx:
            print('[',end='')
            for entry1 in entry:
                if entry1==mtx[0][2] or entry1==mtx[1][2]:
                    print('%f' % entry1,';')
                elif entry1==mtx[2][2]:
                    print('%f' % entry1,';]')
                else:
                    print('%f'% entry1,',',end='')
        cv.waitKey(500)
    cv.destroyAllWindows()
        

    def on_btn1_3_click(self):
        # cboxImgNum to access to the ui object
        current = self.cboxImgNum.currentText()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        path = './images/CameraCalibration/'+repr(int(current))+'.bmp'
        images = glob.glob('./images/CameraCalibration/'+repr(int(current))+'.bmp')
        
        i=1
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv.drawChessboardCorners(img, (11,8), corners2,ret)
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
                R_matrix,J = cv.Rodrigues(rvecs[0]) 
                Extrinsic = np.hstack((R_matrix,tvecs[0]))
                print('[',end='')
                for entry in Extrinsic:
                    for entry1 in entry:
                        if entry1==Extrinsic[0][3] or entry1==Extrinsic[1][3]:
                            print(entry1,';')
                        elif entry1 == Extrinsic[2][-1]:
                            print(entry1,';]')
                        else:
                            print(entry1,',',end='')

    def on_btn1_4_click(self):
        # cboxImgNum to access to the ui object
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('./images/CameraCalibration/*.bmp')
        i=1
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv.drawChessboardCorners(img, (11,8), corners2,ret)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        print(i)
        i=i+1
        print('[',end='')
        for entry in dist:
            for entry1 in entry:
                if entry1==entry[4]:
                    print(entry1,']')
                else:
                    print(entry1,',',end='')

    def on_btn2_1_click(self):
        def draw(img, corners, imgpts):
            imgpts = np.int32(imgpts).reshape(-1,2)
            
            # draw ground floor in green
            img = cv.drawContours(img, [imgpts[:4]],-1,(0,0,255),10)
            
            # draw pillars in blue color
            for i,j in zip(range(4),range(4,8)):
                img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),10)
            
            # draw top layer in red color
            img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),10)
            return img
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        axis = np.float32([[0,0,0], [0,2,0], [2,2,0], [2,0,0],
                   [0,0,-2],[0,2,-2],[2,2,-2],[2,0,-2] ])
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        for fname in glob.glob('./images/2_1/*.bmp'):
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        V_img=[]
            
        for fname in glob.glob('./images/2_1/*.bmp'):
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)
            
            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                
                # Find the rotation and translation vectors.
                x,rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist)
                
                # project 3D points to image plane
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
                
                img = draw(img,corners2,imgpts)
                V_img.append(img)
        h, w, l = V_img[0].shape
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        videoWriter = cv.VideoWriter('./images/v0.avi', 0x7634706d, 2, (w,h)) 
        for i in range(0,5):
            videoWriter.write(V_img[i])


        capture = cv.VideoCapture("./images/v0.avi")

        if capture.isOpened():
            while True:
                ret, prev = capture.read()
                if ret==True:
                    cv.imshow('video', prev)
                else:
                    break
                if cv.waitKey(20)==27:
                    break

        # cap = cv.VideoCapture('./images/v0.mp4')
        # print(type(cap))        
        # ret, frame = cap.read()
        # g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # cv.imshow('frame',videoWriter)
        videoWriter.release()
        cv.destroyAllWindows()

    def on_btn3_1_click(self):
        # edtAngle, edtScale. edtTx, edtTy to access to the ui object
        Angle = self.edtAngle.text()
        Scale = self.edtScale.text()
        Tx = float(self.edtTx.text())
        Ty = float(self.edtTy.text())
        img = cv.imread('./images/OriginalTransform.png')
        H = np.float32([[1,0,Tx],[0,1,Ty]])
        rows,cols = img.shape[:2]
        res = cv.warpAffine(img,H,(rows,cols))
        # rotate & Scale]
        rows,cols = res.shape[:2]
        M = cv.getRotationMatrix2D((130+Tx,125+Ty),float(Angle),float(Scale))
        res = cv.warpAffine(res,M,(rows,cols))
        
        cv.imshow('img',res)

    def on_btn3_2_click(self):
        imgpoints=[]
        objpoints = [[20,20],[450,20],[450,450],[20,450]]
        def draw_circle(event,x,y,flags,param):
            if event == cv.EVENT_LBUTTONDOWN:
                nonlocal imgpoints
                imgpoints.append([x,y])

                if len(imgpoints)==4:
                    pts1 = np.float32(imgpoints)
                    pts2 = np.float32(objpoints)
                    M = cv.getPerspectiveTransform(pts1,pts2)
                    dst = cv.warpPerspective(img,M,(430,430))

                    cv.imshow('new image', dst)

     
        img = cv.imread('./images/OriginalPerspective.png')
        cv.namedWindow('image')
        cv.imshow('image',img)
        cv.setMouseCallback('image',draw_circle)
        

    def on_btn4_1_click(self):
        imgL = cv.imread('./images/imL.png',0)
        imgR = cv.imread('./images/imR.png',0)
        stereo = cv.StereoSGBM_create(numDisparities=48, blockSize=3, disp12MaxDiff=0) #0~47/window size:3x3, block size>5?
        disparity = stereo.compute(imgL,imgR)
        # res = cv.convertScaleAbs(disparity) 
        # res = cv.cvtColor(disparity, cv.COLOR_BGR2RGB)
        normalizedImg = np.zeros((800, 800))
        normalizedImg = cv.normalize(disparity, normalizedImg, 0, 255, cv.NORM_MINMAX,cv.CV_8U)
        cv.imshow('Without L-R Disparity check',normalizedImg)

        # plt.imshow(disparity,'gray')
        # plt.show()

    def on_btn4_2_click(self):
        imgL = cv.imread('./images/imL.png',0)
        imgR = cv.imread('./images/imR.png',0)
        cv.imshow('o',imgL)
        stereo = cv.StereoSGBM_create(numDisparities=48, blockSize=3, disp12MaxDiff=0) #0~47/window size:3x3, block size>5?
        disparity = stereo.compute(imgL,imgR)
        stereo_with = cv.StereoSGBM_create(numDisparities=48, blockSize=3, disp12MaxDiff=2)
        disparity_with = stereo_with.compute(imgL, imgR)
        # res = cv.convertScaleAbs(disparity) 
        # res = cv.cvtColor(disparity, cv.COLOR_BGR2RGB)
        normalizedImg = np.zeros((800, 800))
        normalizedImg = cv.normalize(disparity, normalizedImg, 0, 255, cv.NORM_MINMAX,cv.CV_8U)
        cv.imshow('Without L-R Disparity',normalizedImg)
        normalizedImg_with = np.zeros((800, 800))
        normalizedImg_with = cv.normalize(disparity_with, normalizedImg_with, 0, 255, cv.NORM_MINMAX,cv.CV_8U)
        cv.imshow('With L-R Disparity',normalizedImg_with)
        diff = cv.absdiff(normalizedImg, normalizedImg_with)
        (x,y) = np.where(diff>0)

        backtorgb = cv.cvtColor(normalizedImg_with,cv.COLOR_GRAY2RGB)
        for i in range(len(x)):
            backtorgb[x[i],y[i]]=(0, 0, 255)
        cv.imshow('Mark the diff.', backtorgb)
        # print(type(normalizedImg_with[0,0]))

        # normalizedImg_with[x,y] = (0,0,255)
        # cv.imwrite('./images/temp.jpg',diff)
        # mask = cv.imread('./images/temp.jpg')
        # mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        # th = 1
        # imask = mask>th
        # canvas = np.zeros_like(normalizedImg_with, np.uint8)
        # canvas[imask] = normalizedImg_with[imask]
        # canvas.setTo(new Scalar(0,0,255))
        # print(canvas)
        # output = cv.bitwise_and(normalizedImg_with, normalizedImg_with, mask = canvas)
        # colors = {"red": [0.1,0.,0.], "blue": [0.,0.,0.1]}
        # colored_mask = np.multiply(mask, colors["red"])
        # normalizedImg_with = normalizedImg_with+colored_mask
        # cv.imshow('i',output)
    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
