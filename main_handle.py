from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QPixmap, QImage, QBrush
from origin_gui import Ui_MainWindow
from straightener import Straighten_img, Straighten_img_black, Straighten_img_white
import cv2 
import numpy as np
import imagej
import os.path as osp
import ntpath
import AI_straighten.model as model
from AI_straighten.visualize_for_app import predictStraighten
from Straighten_karyotype.Karyotyping_to_Chromosomes_bio_def import extract_chromosome
from Straighten_karyotype.transparent_img_white import rewrite_image


class MAIN_HANDLE(Ui_MainWindow):
    # list_img =[]
    def __init__(self, mainwindow, ij, model):
        self.setupUi(mainwindow)
        self.ij = ij
        self.model = model
        self.polygon_width = 30
        self.imgIdx = 0
        self.ai_imgs = []
        self.his_dir = ''
        self.list_imgs =[]
        self.list_img = []
        self.img_batch_count = 1
        self.save_dir = None
        self.save_imgs = []
        self.current_tool =[]
        self.save_name = []
        self.max_img_show_batch = 0

        self.btn_select.clicked.connect(self.getImage)
        self.btn_select_2.clicked.connect(self.getKaryoImage)
        self.stackedWidget.setCurrentWidget(self.page)
        self.btn_runMenu.clicked.connect(lambda: self.changePage(2))
        self.btn_startMenu.clicked.connect(lambda: self.changePage(1))
        self.btn_runMenu.hide()

        self.lb_karyotype_img.setPixmap(QPixmap('./Item/chromosome.png').scaled(921,731))
        

        self.stackedWidget_3.setCurrentWidget(self.page_ai)
        self.btn_change_to_draw.clicked.connect(lambda: self.changeToDraw(3))
        self.stackedWidget_2.setCurrentWidget(self.page_normal)


        self.btn_straightener.clicked.connect(lambda: self.changeToDraw(4))



        self.frame_straightener_options.hide()
        self.slider_line_width.setMinimum(20)
        self.slider_line_width.setMaximum(50)
        self.slider_line_width.setValue(30)
        self.slider_line_width.valueChanged.connect(lambda: self.slideIt())
        self.lcd_width.display(30)


        self.btn_nextImg.clicked.connect(lambda: self.changeImg(1))
        self.btn_previousImg.clicked.connect(lambda: self.changeImg(0))

        self.btn_aivstool.clicked.connect(lambda: self.changeAIvsTool())
        self.btn_aivstool.hide()

        self.btn_save.clicked.connect(lambda: self.saveImages())
        
    def getImage(self):
        if self.stackedWidget_4.currentWidget() == self.page_select_karyogram:
            self.stackedWidget_4.setCurrentWidget(self.page_select_singlechro)
            self.list_imgs = []

        imgAddrs,_ = QtWidgets.QFileDialog.getOpenFileNames(caption='Chọn ảnh bạn muốn làm thẳng',directory= self.his_dir ,filter='(*.png *.tiff *.jpg *.jpeg)')
        if len(imgAddrs) > 0:
            self.his_dir = ntpath.split(imgAddrs[0])[0]
        # if len(self.list_img) + len(imgAddrs) < 50:
        self.list_imgs.extend(imgAddrs)
        self.list_imgs.sort()
        
        iconsize =QSize(100,100)

        if len(self.list_imgs)>0:
            self.btn_runMenu.show()
            self.lb_karyotype_img.hide()

        self.list_imgLoad.clear()
        
        for idx in range(len(imgAddrs)):
            icon = QtGui.QIcon(imgAddrs[idx])
            item = QtWidgets.QListWidgetItem(icon, imgAddrs[idx])
            size = QtCore.QSize()
            size.setHeight(120)
            size.setWidth(400)
            item.setSizeHint(size)
            self.list_imgLoad.setIconSize(iconsize)
            self.list_imgLoad.addItem(item)
            

        self.list_imgLoad.itemDoubleClicked.connect(lambda: self.deleteItem())
        self.lb_process.setText(f'{len(self.list_imgs)} image(s) have been chosen.')

    def getKaryoImage(self):
        if self.stackedWidget_4.currentWidget() == self.page_select_singlechro:
            self.stackedWidget_4.setCurrentWidget(self.page_select_karyogram)
            self.list_imgs = []

        self.imgAddr,_ = QtWidgets.QFileDialog.getOpenFileName(caption='Chọn ảnh bạn muốn làm thẳng',directory= self.his_dir ,filter='(*.png *.tiff *.jpg *.jpeg)')
        self.lb_karyotype_img.setPixmap(QPixmap(self.imgAddr).scaled(921,731))
        self.lb_karyotype_img.show()
        if self.imgAddr:
            self.btn_runMenu.show()
        
        self.lb_process.setText('')
    
    def changePage(self, idx):
        if idx == 1:
            self.stackedWidget.setCurrentWidget(self.page)
            self.imgIdx = 0
            self.ai_imgs = []
            self.his_dir = ''
            self.list_imgs =[]
            self.list_img = []
            self.img_batch_count = 1
            self.save_dir = None
            self.save_imgs = []
            self.current_tool =[]
            self.save_name = []
            self.max_img_show_batch = 0


            self.lb_karyotype_img.setPixmap(QPixmap('./Item/chromosome.png').scaled(921,731))
            self.stackedWidget_4.setCurrentWidget(self.page_select_karyogram)
            print(1)
            self.btn_runMenu.hide()
            self.lb_process.setText('')
            
        elif idx ==2 and self.stackedWidget_4.currentWidget() == self.page_select_singlechro:
            
            self.list_img = self.list_imgs[20* (self.img_batch_count-1): (20 * self.img_batch_count if 20 * self.img_batch_count < len(self.list_imgs) else len(self.list_imgs))]
            # self.img_batch_count += 1
            self.ai_imgs =  predictStraighten(self.model, self.list_img)
            self.lb_img_normal.setPixmap(QPixmap(self.list_img[self.imgIdx]).scaled(551, 551))
            self.page_draw.painter.setBackground(QBrush(QPixmap(self.list_img[self.imgIdx]).scaled(551, 551)))
            self.page_draw.painter.eraseRect( QRect(QPoint(0,0), QSize(551,551)))

            current_ai = self.ai_imgs[self.imgIdx]
            pixmapx = numpyToQpixmap(current_ai)
            self.lb_ai_img.setPixmap(pixmapx.scaled(551,551))

            #######
            self.stackedWidget.setCurrentWidget(self.page_2)
            self.btn_startMenu.hide()

            self.lb_process.setText(f'{self.max_img_show_batch + (self.img_batch_count-1)*20}/{len(self.list_imgs)} have been straighten.')


        elif idx ==2 and self.stackedWidget_4.currentWidget() == self.page_select_karyogram:
            self.black_chro, self.white_chro, self.class_box, self.background = extract_chromosome(self.imgAddr)
            self.ai_imgs, self.out_white = predictStraighten(self.model, np.asarray([self.black_chro, self.white_chro]), phase='karyogram')
            self.lb_img_normal.setPixmap(numpyToQpixmap(self.black_chro[self.imgIdx]).scaled(551,551))
            self.page_draw.painter.setBackground(QBrush(numpyToQpixmap(self.black_chro[self.imgIdx]).scaled(551, 551)))
            self.page_draw.painter.eraseRect( QRect(QPoint(0,0), QSize(551,551)))
            
            self.lb_ai_img.setPixmap(numpyToQpixmap(self.ai_imgs[self.imgIdx]).scaled(551,551))

            #######
            self.btn_save.hide()
            self.stackedWidget.setCurrentWidget(self.page_2)

            self.lb_process.setText(f'{self.max_img_show_batch}/{len(self.black_chro)} have been straighten.')

    def changeToDraw(self, idx): 
        if idx == 3:
            if self.stackedWidget_2.currentWidget() == self.page_normal:
                self.btn_change_to_draw.setText('Normal')
                self.stackedWidget_2.setCurrentWidget(self.page_draw)
                self.frame_straightener_options.show()
            else:
                self.btn_change_to_draw.setText('Tool')
                self.stackedWidget_2.setCurrentWidget(self.page_normal)
                self.frame_straightener_options.hide()
        elif idx == 4:
            if len(self.page_draw.x_points) >=2 and self.stackedWidget_4.currentWidget() == self.page_select_singlechro: 
                xpoints = [ int(x * 256 / 551) for x in self.page_draw.x_points ]
                ypoints = [ int(x * 256 / 551) for x in self.page_draw.y_points ]
                self.current_tool = Straighten_img(self.ij, self.list_img[self.imgIdx], xpoints, ypoints, self.polygon_width)
                self.lb_tool.setPixmap(numpyToQpixmap(self.current_tool).scaled(551,551))
                self.stackedWidget_3.setCurrentWidget(self.page_tool)
                self.btn_aivstool.show()
            
            elif len(self.page_draw.x_points) >=2 and self.stackedWidget_4.currentWidget() == self.page_select_karyogram:   
                xpoints = [ int(x * 256 / 551) for x in self.page_draw.x_points ]
                ypoints = [ int(x * 256 / 551) for x in self.page_draw.y_points ]
                self.current_tool = Straighten_img_black(self.ij, self.black_chro[self.imgIdx], xpoints, ypoints, self.polygon_width)
                self.lb_tool.setPixmap(numpyToQpixmap(self.current_tool).scaled(551,551))
                self.current_tool_white = Straighten_img_white(self.ij, self.white_chro[self.imgIdx], xpoints, ypoints, self.polygon_width)
                self.stackedWidget_3.setCurrentWidget(self.page_tool)
                self.btn_aivstool.show()

    def deleteItem(self):
        row = self.list_imgLoad.currentRow()
        # print(row)
        del self.list_imgs[row]
        self.list_imgLoad.takeItem(row)

        self.lb_process.setText(f'{len(self.list_imgs)} image(s) have been chosen.')

    def slideIt(self):
        self.polygon_width = self.slider_line_width.value()
        self.lcd_width.display(self.slider_line_width.value())
        
    def changeImg(self, data):
        x=0
        if self.stackedWidget_3.currentWidget() == self.page_tool:
            self.ai_imgs[self.imgIdx] = self.current_tool
            if self.stackedWidget_4.currentWidget() == self.page_select_karyogram:
                self.out_white[self.imgIdx] = self.current_tool_white
        if self.stackedWidget_4.currentWidget() == self.page_select_singlechro:
            if data == 1 and self.imgIdx < len(self.list_img)-1:
                self.imgIdx += 1
                if self.imgIdx >self.max_img_show_batch:
                    self.max_img_show_batch += 1
                self.btn_startMenu.hide()
                # print(self.imgIdx)
                # print(self.max_img_show_batch)
                
            elif data == 0 and self.imgIdx > 0:
                self.imgIdx -= 1
                
            elif data == 1 and self.imgIdx == len(self.list_img)-1 : #and len(self.list_img) == 20
                # self.imgIdx += 1
                self.saveImages()
                self.btn_startMenu.hide()
                # print((self.img_batch_count-1) * 20 + self.imgIdx + 1)
                if (self.img_batch_count-1) * 20 + self.imgIdx + 1 == len(self.list_imgs):
                    self.changePage(1)
                    x = 1
                else:
                    self.imgIdx = 0
                    self.max_img_show_batch = 0

                    if self.img_batch_count  < int(len(self.list_imgs) / 20 + 1):
                        self.img_batch_count += 1
                    # print(self.img_batch_count)
                    # print(20* (self.img_batch_count-1))
                    # print(20 * self.img_batch_count if 20 * self.img_batch_count < len(self.list_imgs) else len(self.list_imgs))
                    self.list_img = self.list_imgs[20* (self.img_batch_count-1): (20 * self.img_batch_count if 20 * self.img_batch_count < len(self.list_imgs) else len(self.list_imgs))]
                    self.ai_imgs = predictStraighten(self.model, self.list_img)

            if x == 0:
                #### stackWidget_2
                self.lb_img_normal.setPixmap(QPixmap(self.list_img[self.imgIdx]).scaled(551, 551))
                self.page_draw.painter.setBackground(QBrush(QPixmap(self.list_img[self.imgIdx]).scaled(551, 551)))
                self.page_draw.painter.eraseRect( QRect(QPoint(0,0), QSize(551,551)))
                

                #### stackWidget_3
                self.lb_ai_img.setPixmap(numpyToQpixmap(self.ai_imgs[self.imgIdx]).scaled(551,551))
            
            self.lb_process.setText(f'{self.max_img_show_batch + (self.img_batch_count - 1) * 20}/{len(self.list_imgs)} image(s) have been straightened.')

        else: 
            if data == 1 and self.imgIdx < len(self.ai_imgs)-1:
                self.imgIdx += 1
                self.max_img_show_batch += 1
                self.btn_save.show()
                self.btn_startMenu.hide()
                
            elif data == 0 and self.imgIdx > 0:
                self.imgIdx -= 1
                
            elif data == 1 and self.imgIdx == len(self.ai_imgs)-1:
                # self.new_karyo_img = rewrite_image(self.out_white, self.background, self.class_box)
                # self.imgIdx += 1
                self.saveImages()
                self.imgIdx = 0
                self.max_img_show_batch = 0
                self.ai_imgs = []
                self.changePage(1)
                self.btn_startMenu.hide()

            self.frame_straightener_options.hide()
            self.page_draw.x_points = []
            self.page_draw.y_points= []
            
            #### stackWidget_2
            self.lb_img_normal.setPixmap(numpyToQpixmap(self.black_chro[self.imgIdx]).scaled(551, 551))
            self.page_draw.painter.setBackground(QBrush(numpyToQpixmap(self.black_chro[self.imgIdx]).scaled(551, 551)))
            self.page_draw.painter.eraseRect( QRect(QPoint(0,0), QSize(551,551)))
            

            #### stackWidget_3
            if self.ai_imgs != []:
               self.lb_ai_img.setPixmap(numpyToQpixmap(self.ai_imgs[self.imgIdx]).scaled(551,551))

            #### label process
            self.lb_process.setText(f'{self.max_img_show_batch}/{len(self.black_chro)} have been straighten.')


        self.frame_straightener_options.hide()
        self.page_draw.x_points = []
        self.page_draw.y_points= []
        ##### pages
        self.stackedWidget_2.setCurrentWidget(self.page_normal)
        self.stackedWidget_3.setCurrentWidget(self.page_ai)

        ##### buttons
        self.btn_change_to_draw.setText("Tool")
        self.btn_aivstool.hide()

        
            
    def changeAIvsTool(self):
        if self.stackedWidget_3.currentWidget() == self.page_ai:
            self.stackedWidget_3.setCurrentWidget(self.page_tool)
            self.btn_aivstool.setText('Ai')
        else:
            self.stackedWidget_3.setCurrentWidget(self.page_ai)
            self.btn_aivstool.setText('Tool')
    
    def saveImages(self):
        if self.stackedWidget_4.currentWidget() == self.page_select_singlechro:
            if self.stackedWidget_3.currentWidget() == self.page_tool:
                self.ai_imgs[self.imgIdx] = self.current_tool

            if self.save_dir == None:
                self.save_dir = QtWidgets.QFileDialog.getExistingDirectory(caption='Chọn file bạn muốn lưu')
            for idx in range(self.max_img_show_batch +1):
                save_imgNames = osp.join(self.save_dir, ntpath.split(self.list_img[idx])[1])
                print(save_imgNames)
                # print(self.max_img_show_batch)
                # cv2.imwrite(save_imgNames, self.ai_imgs[idx])
        else:
            self.save_dir = QtWidgets.QFileDialog.getExistingDirectory(caption='Chọn file bạn muốn lưu')
            save_imgNames = osp.join(self.save_dir, 'new'+ ntpath.split(self.imgAddr)[1])
            # print(save_imgNames)
            # new_karyotyping_img = rewrite_image(self.out_white, self.background, self.class_box)
            # cv2.imwrite(save_imgNames, new_karyotyping_img)
        self.btn_startMenu.show()
    

def numpyToQpixmap(np_Img):
    cv_img = np_Img.astype(np.uint8)
    if len(cv_img.shape)<3:
        frame = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    else:
        frame = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = cv_img.shape[:2]
    bytesPerLine = 3 * w
    qimage = QImage(frame.data, w, h, bytesPerLine, QImage.Format.Format_RGB888) 
    pixmapx = QPixmap.fromImage(qimage)
    return pixmapx

def preprocess_img(img):
    img = img.astype(np.uint8)
    # cv2.imwrite('test_straigh.jpg', img)
    # connectedComponents(img)
    for i in range(4):
        _, pre_thresh = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
        _, pre_thresh2 = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)
        # cv2.imwrite('test_straigh.jpg', pre_thresh)
        thresh = cv2.morphologyEx(pre_thresh, cv2.MORPH_CLOSE, np.ones(shape=(3, 3), dtype=np.uint8))
        # cv2.imwrite('test_straigh.jpg', pre_thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  
        
        contours = list(contours)
        if len(contours) > 1:
            max_h = 0
            idx = 0
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.
                # cv2.imwrite('test_straigh.jpg', img)
                if h > max_h and h < 255:
                    idx = i
            new_img = np.ones_like(thresh) * 255
            x, y, w, h = cv2.boundingRect(contours[idx])
            new_img[y:y+h-2,x:x+w] =img[y:y+h-2,x:x+w]
            img = new_img
        pre_thresh2 = pre_thresh2.clip(0,1)
        img = img * pre_thresh2
        img = img + pre_thresh

    return img



if __name__ == "__main__":
    import sys
    import AI_straighten.config as config
    import torch

    ########## imagej
    ij = imagej.init('2.5.0', mode='interactive')

    ########## model straighten
    model_predict = model.get_model(config.TRAINING_CONDITION)
    checkpoint = torch.load('./weight/straightening.pth', map_location=torch.device('cpu'))
    model_predict.load_state_dict(checkpoint)
    model_predict.eval()


    ############## Run App
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MAIN_HANDLE(MainWindow, ij, model_predict)

    MainWindow.show()
    sys.exit(app.exec_())
