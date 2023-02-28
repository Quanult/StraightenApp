import sys
from PyQt5.QtWidgets import QApplication, QWidget,  QVBoxLayout, QPushButton, QGraphicsSceneMouseEvent
from PyQt5.QtCore import Qt, QPoint, QRect,QLineF, QSize
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush


class Interactive(QWidget):
	def __init__(self, image_path = None):
		super().__init__()
		# self.parent = parent
		self.window_width, self.window_height = 551, 551
		self.setMinimumSize(self.window_width, self.window_height)

		layout = QVBoxLayout()
		self.setLayout(layout)
		if image_path == None :
			self.img_path = 'chromosome.png'
		else:
			self.img_path = image_path
		
		self.pix = QPixmap(QPixmap(self.rect().size()))
		self.pix.fill(Qt.white)
		
		
		self.begin, self.end = QPoint(), QPoint()	
		self.painter = QPainter(self.pix)
		# self.background_img = QPixmap('img_test.jpg').scaled(551, 551)
		# self.background = QBrush(self.background_img)
		# # self.background.set
		# self.painter.setBackground(self.background)
		# self.button = QPushButton()
		# self.button.setGeometry(QtCore.QRect(600, 600, 93, 28))
		self.x_points = []
		self.y_points = []

	def paintEvent(self, event):
		Painter = QPainter(self)
		Painter.drawPixmap(QPoint(), self.pix)
		for idx in range(len(self.x_points)):
			if  idx < (len(self.x_points)-1):
				linePen = QPen(Qt.yellow, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
				self.painter.setPen(linePen)
				line_begin = QPoint(self.x_points[idx],self.y_points[idx])
				line_end = QPoint(self.x_points[idx+1],self.y_points[idx+1])
				line = QLineF(line_begin, line_end)
				self.painter.drawLine(line)
				
			rect_begin = QPoint(self.x_points[idx]-3,self.y_points[idx]-3 )
			rect = QRect(rect_begin, QSize(6,6))
			linePen = QPen(Qt.red, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
			self.painter.setPen(linePen)
			self.painter.drawRect(rect.normalized())

	def mousePressEvent(self, event):
		if event.buttons() & Qt.LeftButton:
			self.end = event.pos()
			if self.end.x() <= 551 and self.end.y() <= 551:
				self.x_points.append(self.end.x())
				self.y_points.append(self.end.y())
			self.update()
		if event.buttons() & Qt.RightButton:
			if len(self.x_points) >=1:
				self.x_points.pop(-1)
				self.y_points.pop(-1)
			erase_rect = QRect(QPoint(0,0), QSize(551,551))
			
			self.painter.eraseRect(erase_rect)
			# self.painter.fillRect(erase_rect, self.pixx)
			for idx in range(len(self.x_points)):
				if  idx < (len(self.x_points)-1):
					linePen = QPen(Qt.yellow, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
					self.painter.setPen(linePen)
					line_begin = QPoint(self.x_points[idx],self.y_points[idx])
					line_end = QPoint(self.x_points[idx+1],self.y_points[idx+1])
					line = QLineF(line_begin, line_end)
					self.painter.drawLine(line)
					
				rect_begin = QPoint(self.x_points[idx]-3,self.y_points[idx]-3 )
				rect = QRect(rect_begin, QSize(6,6))
				linePen = QPen(Qt.red, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
				self.painter.setPen(linePen)
				self.painter.drawRect(rect.normalized())
			self.update()

	def mouseReleaseEvent(self, event):
		if event.button() & Qt.LeftButton:
			self.begin = self.end
			self.update()


if __name__ == '__main__':
	
	app = QApplication(sys.argv)
	app.setStyleSheet('''
		QWidget {
			font-size: 30px;
		}
	''')
	
	myApp = Interactive()
	myApp.show()

	try:
		sys.exit(app.exec_())
	except SystemExit:
		print('Closing Window...')


