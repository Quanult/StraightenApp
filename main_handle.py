from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QPoint, QRect,QLineF, QSize
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QBrush
# from interactive import Interactive
from origin_gui import Ui_MainWindow
from straightener import Straighten_img
import cv2 
import numpy as np
import imagej
import os.path as osp
import ntpath

class MAIN_HANDLE(Ui_MainWindow):
    # list_img =[]
    def __init__(self, mainwindow, ij):
        self.setupUi(mainwindow)
        self.ij = ij
        self.polygon_width = 30
        self.imgIdx = 0
        self.ai_imgs = np.zeros((20, 256, 256, 3))
        self.his_dir = ''
        self.list_imgs =[]
        self.list_img = []
        self.img_batch_count = 1
        self.save_dir = None
        self.save_imgs = []
        self.current_tool =[]
        self.save_name = []
        self.max_img_show_batch = 1

        self.btn_select.clicked.connect(self.getImage)
        self.stackedWidget.setCurrentWidget(self.page)
        self.btn_runMenu.clicked.connect(lambda: self.changePage(2))
        self.btn_startMenu.clicked.connect(lambda: self.changePage(1))
        self.btn_runMenu.hide()
        

        self.stackedWidget_3.setCurrentWidget(self.page_ai)
        self.btn_change_to_draw.clicked.connect(lambda: self.changePage(3))
        self.stackedWidget_2.setCurrentWidget(self.page_normal)


        self.btn_straightener.clicked.connect(lambda: self.changePage(4))



        self.frame_straightener_options.hide()
        self.slider_line_width.setMinimum(20)
        self.slider_line_width.setMaximum(50)
        self.slider_line_width.setValue(30)
        self.slider_line_width.valueChanged.connect(lambda: self.slideIt())
        self.lcd_width.display(30)


        self.btn_nextImg.clicked.connect(lambda: self.changeImg(1))
        self.btn_previousImg.clicked.connect(lambda: self.changeImg(0))

        self.btn_aivstool.clicked.connect(lambda: self.changeAIvsTool())

        self.btn_save.clicked.connect(lambda: self.saveImages())
        
    def getImage(self):
        imgAddrs,_ = QtWidgets.QFileDialog.getOpenFileNames(caption='Chọn ảnh bạn muốn làm thẳng',directory= self.his_dir ,filter='(*.png *.tiff *.jpg *.jpeg)')
        if len(imgAddrs) > 0:
            self.his_dir ,_ = ntpath.split(imgAddrs[0])
        # if len(self.list_img) + len(imgAddrs) < 50:
        self.list_imgs.extend(imgAddrs)
        self.list_img = self.list_imgs[20* (self.img_batch_count-1): (20 * self.img_batch_count if 20 * self.img_batch_count < len(self.list_imgs) else len(self.list_imgs))]
        self.img_batch_count += 1

        iconsize =QSize(100,100)
        if len(self.list_imgs)>0:
            self.btn_runMenu.show()
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

    def changePage(self, idx):
        if idx == 1:
            self.stackedWidget.setCurrentWidget(self.page)
        elif idx ==2 :
            self.stackedWidget.setCurrentWidget(self.page_2)
            self.lb_img_normal.setPixmap(QPixmap(self.list_img[self.imgIdx]).scaled(551, 551))
            self.page_draw.painter.setBackground(QBrush(QPixmap(self.list_img[self.imgIdx]).scaled(551, 551)))
            self.page_draw.painter.eraseRect( QRect(QPoint(0,0), QSize(551,551)))

            current_ai = self.ai_imgs[self.imgIdx]
            pixmapx = numpyToQpixmap(current_ai)
            self.lb_ai_img.setPixmap(pixmapx.scaled(551,551))
         
        elif idx == 3:
            if self.stackedWidget_2.currentWidget() == self.page_normal:
                self.btn_change_to_draw.setText('Normal')
                self.stackedWidget_2.setCurrentWidget(self.page_draw)
                self.frame_straightener_options.show()
            else:
                self.btn_change_to_draw.setText('Tool')
                self.stackedWidget_2.setCurrentWidget(self.page_normal)
                self.frame_straightener_options.hide()
        elif idx == 4:
            if len(self.page_draw.x_points) >=2:
                
                xpoints = [ int(x * 256 / 551) for x in self.page_draw.x_points ]
                ypoints = [ int(x * 256 / 551) for x in self.page_draw.y_points ]
                self.current_tool = Straighten_img(self.ij, self.list_img[self.imgIdx], xpoints, ypoints, self.polygon_width)
                self.lb_tool.setPixmap(numpyToQpixmap(self.current_tool).scaled(551,551))
                self.stackedWidget_3.setCurrentWidget(self.page_tool)

    def deleteItem(self):
        row = self.list_imgLoad.currentRow()
        del self.list_imgs[row]
        self.list_imgLoad.takeItem(row)

    def slideIt(self):
        self.polygon_width = self.slider_line_width.value()
        self.lcd_width.display(self.slider_line_width.value())
        
    def changeImg(self, data):
        if self.stackedWidget_3.currentWidget() == self.page_tool:
            self.ai_imgs[self.imgIdx] = self.current_tool

        if data == 1 and self.imgIdx < len(self.list_img)-1:
            self.imgIdx += 1
            self.max_img_show_batch += 1
        elif data == 0 and self.imgIdx > 0:
            self.imgIdx -= 1
        elif data == 1 and self.imgIdx == 19:
            self.saveImages()
            self.imgIdx = 0
            self.max_img_show_batch += 1
            self.max_img_show_batch = 1

            self.list_img = self.list_img[20* (self.img_batch_count-1): (20 * self.img_batch_count if 20 * self.img_batch_count < len(self.list_img) else len(self.list_img))]
            self.img_batch_count += 1

        
        self.frame_straightener_options.hide()
        self.page_draw.x_points = []
        self.page_draw.y_points= []
        
        #### stackWidget_2
        self.lb_img_normal.setPixmap(QPixmap(self.list_img[self.imgIdx]).scaled(551, 551))
        self.page_draw.painter.setBackground(QBrush(QPixmap(self.list_img[self.imgIdx]).scaled(551, 551)))
        self.page_draw.painter.eraseRect( QRect(QPoint(0,0), QSize(551,551)))
        self.stackedWidget_2.setCurrentWidget(self.page_normal)

        #### stackWidget_3
        self.lb_ai_img.setPixmap(numpyToQpixmap(self.ai_imgs[self.imgIdx]).scaled(551,551))
        self.stackedWidget_3.setCurrentWidget(self.page_ai)
            
    def changeAIvsTool(self):
        if self.stackedWidget_3.currentWidget() == self.page_ai:
            self.stackedWidget_3.setCurrentWidget(self.page_tool)
            self.btn_aivstool.setText('Ai')
        else:
            self.stackedWidget_3.setCurrentWidget(self.page_ai)
            self.btn_aivstool.setText('Tool')
    
    def saveImages(self):
        if self.stackedWidget_3.currentWidget() == self.page_tool:
            self.ai_imgs[self.imgIdx] = self.current_tool

        if self.save_dir == None:
            self.save_dir = QtWidgets.QFileDialog.getExistingDirectory(caption='Chọn file bạn muốn lưu')
        for idx in range(self.max_img_show_batch):
            save_imgNames = osp.join(self.save_dir, path_leaf(self.list_img[idx]))
            cv2.imwrite(save_imgNames, self.ai_imgs[idx])

        
        

    
    
    # def resetChangeImg(self):
        
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

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
        

if __name__ == "__main__":
    import sys

    ij = imagej.init('2.5.0', mode='interactive')


    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MAIN_HANDLE(MainWindow, ij)

    MainWindow.show()
    sys.exit(app.exec_())
