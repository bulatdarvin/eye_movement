from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGraphicsTextItem, QMainWindow, QVBoxLayout, QGridLayout, QPushButton, QHBoxLayout, QGraphicsView, QGraphicsScene,QGraphicsItem, QGraphicsEllipseItem
from PyQt5.QtGui import QPixmap, QColor
import sys
import cv2
from PyQt5.QtCore import pyqtSignal,QTimer, pyqtSlot, Qt, QThread, QRectF, QRect, QPointF, QSizeF
import numpy as np



class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()




class App(QWidget):
    def __init__(self):
        super().__init__()

       # self.setAttribute(Qt.WA_NoSystemBackground)
       # self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')

        timer = QTimer(self)
        timer.timeout.connect(self.animation)
        timer.start(100)
        # create a vertical box layout and add the two labels

        lay = QHBoxLayout(self)
        but = QGridLayout()
        vid = QGridLayout()

        view = QGraphicsView()
        scene = QGraphicsScene()
        scene.setSceneRect(0,0, 300, 300)
        view.setScene(scene)
        pen = QtGui.QPen(Qt.green)

        side = 100

        self.el = QGraphicsEllipseItem(0,0, 20,20)
        self.el.setFlag(QGraphicsItem.ItemIsMovable)

        scene.addItem(self.el)
        for i in range(3):
            for j in range(3):
                r = QRectF(QPointF(i * side, j * side), QSizeF(side, side))
                t = QGraphicsTextItem(str(i)+str(j))
                t.setPos(i*side+side//2, j*side+side//2)
                scene.addItem(t)
                #scene.addText(t, i*side, j*side)
                scene.addRect(r, pen)

        buttons = {}

        vid.addWidget(self.image_label)
        main_but = QPushButton('Click', self)
        main_but.clicked.connect(self.change_wid)
        vid.addWidget(main_but)
        #self.but = but
        #but.addItem(vid)
        lay.addWidget(view)
        lay.addLayout(vid)

        self.setLayout(lay)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()


    def animation(self):
        self.el.moveBy(10, 0)
        self.update()

    def change_wid(self):
        for i in range(3):
            for j in range(3):
                self.but.removeWidget(self.buttons[(i, j)])
                self.buttons[(i, j)] = QPushButton('h %d, j %d' % (i, j))
                # add to the layout

                self.but.addWidget(self.buttons[(i, j)], i, j)
                self.buttons[(i, j)].setMinimumSize(200, 200)

 #   def paintEvent(self, QPaintEvent):
  #      myQPainter = QtGui.QPainter(self)
   #     rect = QRect(100, 150, 250, 25)
    #    myQPainter.drawRect(rect)
     #   myQPainter.drawText(rect, Qt.AlignCenter, "Hello World")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    #w = MyFirstGuiProgram()
    #w.show()
    #a = Draw()
    #a.show()
    sys.exit(app.exec_())