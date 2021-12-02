import cv2
import dlib
import numpy as np
print(cv2.__version__)




def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    try:
        #print(cnts)
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        #print(M)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
           # cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        #cv2.circle(img, (cx, cy), 4, (0, 200, 200), 2)
        return cx, cy
    except:
        return 0, 0
        pass

def middle(point1, point2):
    x = (point1[0] + point2[0]) // 2
    y = (point1[1] + point2[1]) // 2
    return (x, y)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "EDSR_x4.pb"
sr.readModel(path)
sr.setModel("edsr", 4)

cap = cv2.VideoCapture(0)
wind_size = [640, 360]
point = (wind_size[0]//2, wind_size[1]//2)

#ret, img = cap.read()
#thresh = img.copy()

#cv2.namedWindow('image')




def nothing(x):
    pass

def line_point_distance(line, point):
    return np.cross(line[1] - line[0], point - line[0]) / np.linalg.norm(line[1] - line[0])

def movement_control(cap, center, vertical, horizontal, point):
    dst_hor = line_point_distance(vertical, center)
    dst_vert = line_point_distance(horizontal, center)
    p = list(point)
    delta = [0, 0]
    if abs(dst_vert) > 1:
        #print('vert')
        delta[0] += 20 if dst_vert > 0 else -10
        p[0] += delta[0]
    if abs(dst_hor) > 1:
       # print('hor')
        delta[1] += 20 if dst_hor > 0 else -10
        p[1] += delta[1]
    p[0] = p[0] % wind_size[0]
    p[1] = p[1] % wind_size[1]
    #print(delta)
    return p, list(delta)

def threshold_operation(eyes_gray):
    threshold = 60
    _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)  # 1
    thresh = cv2.dilate(thresh, None, iterations=4)  # 2
    thresh = cv2.medianBlur(thresh, 3)  # 3
    thresh = cv2.bitwise_not(thresh)
    return thresh

def lines(shape):
    shape = np.array(shape)
    horizontal_left = np.vstack((shape[36] - 5, shape[39] - 5))
    horizontal_right = np.vstack((shape[42], shape[45]))

    # cv2.line(img, (shape[36][0], shape[36][1] - 5), (shape[39][0], shape[39][1] - 5), (0, 100, 100))
    # cv2.line(img, (shape[36][0], shape[36][1]), (shape[39][0], shape[39][1]), (0, 100, 100))
    # cv2.line(img, (shape[42][0], shape[42][1]), (shape[45][0], shape[45][1]), (0, 100, 100))
    # Vertical

    middle_points = [middle(shape[37], shape[38]),
                     middle(shape[40], shape[41]),
                     middle(shape[43], shape[44]),
                     middle(shape[46], shape[47])]

    vertical_left = middle_points[:2]
    vertical_right = middle_points[2:]
    return vertical_left, vertical_right, horizontal_left, horizontal_right, middle_points


def mask(img, shape, kernel):
    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = eye_on_mask(mask, left, shape)
    mask = eye_on_mask(mask, right, shape)
    mask = cv2.dilate(mask, kernel, 5)
    eyes = cv2.bitwise_and(img, img, mask=mask)
    mask = (eyes == [0, 0, 0]).all(axis=2)
    eyes[mask] = [255, 255, 255]
    return eyes




def blinking(img, mid):
    dst = abs(mid[0][1] - mid[1][1])
    if dst < 3:
        start_time = t.time





global delta
delta = 0


def find_eye(sr, cap, detector, predictor, point):
#while (True):
    start_time = 0
    kernel = np.ones((3, 3), np.uint8)
    ret, img = cap.read()
    #print(img.shape)
    img = cv2.resize(img, (1280//2, 720//2), interpolation=cv2.INTER_AREA)
    #cv2.imshow('eyes', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        eyes = mask(img, shape, kernel)
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        thresh = threshold_operation(eyes_gray)
        #cv2.imshow('t', thresh)
        cx, cy = contouring(thresh[:, 0:mid], mid, img)
        contouring(thresh[:, mid:], mid, img, True)

        ver_l, ver_r, hor_l, hor_r, mid = lines(shape)
        #print(mid[0][1],mid[1][1], shape[36][0],shape[39][0])
        frame = img[mid[0][1]:mid[1][1], shape[36][0]:shape[39][0]]
        frame = sr.upsample(frame)
        frame = cv2.resize(frame, (320, 180), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        cv2.line(img, mid[0], mid[1], (0, 100, 250))
        cv2.line(img, mid[2], mid[3], (0, 100, 250))
        if cx and cy:
         #   cv2.circle(img, (point), 10, (255, 255, 255))
            center = np.array((cx, cy))
            point, delta = movement_control(img, center, np.array(ver_l), hor_l, point)
            return img, frame, point, delta
       # try:

        #except UnboundLocalError:
    return img, img, point, list(point)
        #print('asfasfafasfasf')
        #print(img)
   # cv2.imshow('eyes', img)
        #cv2.imshow('f', frame)
#    if cv2.waitKey(10) & 0xFF == ord('q'):
 #       break
    #cv2.waitKey(1)

#cap.release()
#cv2.destroyAllWindows()

'''
if __name__ =='main':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_68.dat')

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "EDSR_x4.pb"
    sr.readModel(path)
    sr.setModel("edsr", 4)

    cap = cv2.VideoCapture(0)
    wind_size = [640, 360]
    point = (wind_size[0] // 2, wind_size[1] // 2)
    frame = None
    #find_eye(sr, cap, detector, predictor, point, frame)
'''