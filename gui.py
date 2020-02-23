from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QTextEdit
from PySide2.QtGui import QImage, QPainter, QMouseEvent, QPen, QPaintEvent, QFont
from PySide2.QtCore import Qt, QPoint

from data_processing import prepare_image
import torch
import numpy as np
from PIL import Image


NET = torch.load('model/model_CNN.pth')
NET.eval()


class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setGeometry(400, 400, 400, 400)
        self.setWindowTitle('Handwritten digit recognition')

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.init_text()

        self.drawing = False
        self.brush_size = 8
        self.brush_color = Qt.black
        self.last_point = QPoint()

        self.init_btn_clear()
        self.init_btn_recognize()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event: QPaintEvent):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def init_btn_clear(self):
        btn = QPushButton('Clear', self)
        btn.resize(80, 25)
        btn.move(50, 340)
        btn.show()
        btn.clicked.connect(self.clear)

    def clear(self):
        self.image.fill(Qt.white)
        self.text.setText('')
        self.update()

    def init_btn_recognize(self):
        btn = QPushButton('Recognize', self)
        btn.resize(80, 25)
        btn.move(150, 340)
        btn.show()
        btn.clicked.connect(self.recognize)

    def init_text(self):
        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QTextEdit.NoWrap)
        self.text.insertPlainText('')
        font = self.text.font()
        font.setFamily('Rockwell')
        font.setPointSize(25)
        self.text.setFont(font)
        self.text.resize(50, 50)
        self.text.move(266, 324)

    def recognize(self):
        # Convert to image
        image = self.image.convertToFormat(QImage.Format_ARGB32)
        width = image.width()
        height = image.height()
        ptr = image.constBits()
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        im = Image.fromarray(arr[..., :3])
        im.save('test.png')

        # Load image, evaluate net and show result
        img_list = prepare_image('test.png')
        img_tensor = torch.FloatTensor(img_list).unsqueeze(0)
        prediction = torch.argmax(NET(img_tensor)).item()
        self.text.setText(' '+str(prediction))
