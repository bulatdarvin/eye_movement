
from tkinter import *
from main import find_eye
from PIL import Image, ImageTk
import dlib
import cv2
import time
import numpy as np
root = Tk()

global point
global delta

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "EDSR_x4.pb"
sr.readModel(path)
sr.setModel("edsr", 4)

cap = cv2.VideoCapture(0)
global wind_size
wind_size = [640, 360]
point = [wind_size[0]//2, wind_size[1]//2]
print("P", point)

imageFrame = Frame(root, width=1000, height=1000)
imageFrame.grid()
dim = [600, 600]
canvas1 = Canvas(imageFrame, relief = FLAT, background = "#D2D2D2", width = dim[0], height = dim[1])
canvas1.grid(row=0, column=0, pady=(5, 0), sticky='nw')
canvas2 = Canvas(imageFrame, relief = FLAT, background = "#D2D2D2", width = dim[0], height = dim[1])
#canvas2.grid(row=0, column=2, pady=(5, 0), sticky='nw')
#imageFrame.pack()
lmain = Label(imageFrame)
lmain.grid(row=0, column=1)


def clicked(event):
    print("pressed")



def move():


    #print(point, wind_size)
    #_, delta = show_frame(0)
    point = canvas1.coords(circle)
    print(point, delta)
    #if abs(delta[0]) >= dim[0] - 30:
     #   delta[0] = 0
    #if abs(delta[1]) >= dim[1] - 30:
     #   delta[1] = 0
    tmp = delta
    if point[0] + delta[0] > dim[0]:
        print('1')
        delta[0] -= dim[0]
    elif point[0] + delta[0] < 0:
        print('2')
        delta[0] = dim[0]
    if point[1] + delta[1] > dim[1]:
        print('3')
        delta[1] -= dim[1]
    elif point[1] - delta[1] < 0:
        print('4')
        delta[1] = dim[1]
    #point += delta
    #print(delta)
    #point[0] += delta[0]
    #point[1] += point[1]
    #print('after', delta)
    #if point[1] >= 50:
        #print('fasfaf')
        #canvas2.grid_remove()
    canvas1.move(circle, delta[0], delta[1])
    #print('asfasf', delta)
    #delta = tmp
    point = canvas1.coords(circle)
   # x = coordinates[0]
    #y = coordinates[1]
    root.after(100, move)





cap = cv2.VideoCapture(0)
def show_frame(start = 0):
    if start == 0:
        point = [dim[0]//2, dim[1]//2]
        start = 1
    img, frame, point, delta = find_eye(sr, cap, detector, predictor, point)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    #print('show', delta)
    lmain.after(100, show_frame)
    return point, delta

#point, delta = show_frame(0)

#canvas1.pack()
w = dim[0]//3
h = dim[1]//3
t = 10
for i in range(3):
    for j in range(3):
        buttonBG = canvas1.create_rectangle(i*w, j*h, (i+1)*w, (j+1)*h, fill="grey"+str(t), outline="grey"+str(t))
        t += 10
        buttonTXT = canvas1.create_text(i*w+w//2, j*h+h//2, text="click"+str(i)+str(j))
        canvas1.tag_bind(buttonBG, "Button-"+str(i)+'-'+str(j), clicked)  ## when the square is clicked runs function "clicked".
        canvas1.tag_bind(buttonTXT, "Button-"+str(i)+'-'+str(j), clicked)

t = 10
for i in range(3):
    for j in range(3):
        buttonBG = canvas2.create_rectangle(i*w, j*h, (i+1)*w, (j+1)*h, fill="grey"+str(t), outline="grey"+str(t))
        t += 10
        buttonTXT = canvas2.create_text(i*w+w//2, j*h+h//2, text="click"+str(i)+str(j))
        canvas2.tag_bind(buttonBG, "New-"+str(i)+'-'+str(j), clicked)  ## when the square is clicked runs function "clicked".
        canvas2.tag_bind(buttonTXT, "New-"+str(i)+'-'+str(j), clicked)


circle = canvas1.create_oval(10, 5, 50, 50, fill="blue")
root.after(100, move)

root.mainloop()


class MyVideoCapture:

    def __init__(self, video_source=0, width=None, height=None):

        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = width
        self.height = height

        # Get video source width and height
        if not self.width:
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # convert float to int
        if not self.height:
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # convert float to int

        self.ret = False
        self.frame = None

    def process(self):
        ret = False
        frame = None

        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.ret = ret
        self.frame = frame

    def get_frame(self):
        self.process()  # later run in thread
        return self.ret, self.frame

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class tkCamera(Frame):

    def __init__(self, window, video_source=0, width=None, height=None):
        super().__init__(window)

        self.window = window

        # self.window.title(window_title)
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source, width, height)

        self.canvas = Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor='center', expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update_widget()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update_widget(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        self.window.after(self.delay, self.update_widget)


class App:

    def __init__(self, window, window_title, video_sources):
        self.window = window

        self.window.title(window_title)

        self.vids = []

        for source in video_sources:
            vid = tkCamera(window, source, 400, 300)
            vid.pack()
            self.vids.append(vid)

        # Create a canvas that can fit the above video source size

        self.window.mainloop()


if __name__ == '__main__':
    sources = [
        0,
        # 'https://imageserver.webcamera.pl/rec/krupowki-srodek/latest.mp4',
        # 'https://imageserver.webcamera.pl/rec/skolnity/latest.mp4',
        'https://imageserver.webcamera.pl/rec/krakow4/latest.mp4',
    ]

    # Create a window and pass it to the Application object
    App(tkinter.Tk(), "Tkinter and OpenCV", sources)