from ultralytics import YOLO
import cv2
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils import face_utils
import numpy as np
import dlib
from threading import Thread
import pyttsx3
import imutils
import os

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def sound_alarm(path):
    # play an alarm sound
    pyttsx3.pyttsx3(path)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    # return the eye aspect ratio
    return ear

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "'+msg+'"'
        os.system(s)
        saying = False

# Define constants for drowsiness detection
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
YAWN_THRESH = 20
BLINK_COUNTER = 0
alarm_status = False
alarm_status2 = False
saying = False
EYES_CLOSED = False

# Define paths
video_path = r"C:\\Users\\Chapri 007\\OneDrive\\Desktop\\Drowsiness-Detection\\Drowsiness-Detection\\SIDE_VEIWthree.mp4"
model_path = r"C:\\Users\\Chapri 007\\OneDrive\\Desktop\\Drowsiness-Detection\\Drowsiness-Detection\\best.pt"

# Video capture for pose estimation
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("myvideo.mp4", fourcc, fps, (width, height))

# Load DriPE pose estimation model
model = YOLO(model_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pose estimation
    results = model(source=frame, show=False, conf=0.7, save=False)
    x1, y1, x2, y2 = 251, 180, 368, 380

    for result in results:
        keypoints = result.keypoints.xy
        boxes = result.boxes
        img = result.orig_img
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for n, i in enumerate(keypoints[0]):
            x, y = int(i[0]), int(i[1])
            left_wrist = keypoints[0][9]
            right_wrist = keypoints[0][10]
            lwx, lwy = int(left_wrist[0]), int(left_wrist[1])
            rwx, rwy = int(right_wrist[0]), int(right_wrist[1])
            cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

            # Check if wrists are outside the rectangle
            if n == 9 or n == 10:
                if not (x1 < lwx < x2 and y1 < lwy < y2):
                    print("Danger")
                    cv2.circle(img, (lwx, lwy), 1, (0, 0, 255), 2)
                    cv2.putText(img, "Left Wrist is not on Steering", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if not (x1 < rwx < x2 and y1 < rwy < y2):
                    print("Danger")
                    cv2.circle(img, (rwx, rwy), 1, (0, 0, 255), 2)
                    cv2.putText(img, "Right Wrist is not on Steering", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Drowsiness detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        mouth = shape[48:68]
        leftEye = shape[42:48]
        rightEye = shape[36:42]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        # Yawn detection
        mouthHull = cv2.convexHull(mouth)
        distance = dist.euclidean(mouth[14], mouth[18])
        
        if distance > YAWN_THRESH:
            cv2.putText(frame, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                t = Thread(target=alarm, args=('Take some fresh air, sir',))
                t.deamon = True
                t.start()
        else:
            alarm_status2 = False
        
        # Eye status
        if ear < EYE_AR_THRESH:
            cv2.putText(frame, "Left Eye: Closed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Left Eye: Open", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if rightEAR < EYE_AR_THRESH:
            cv2.putText(frame, "Right Eye: Closed", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Right Eye: Open", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blink Counter: {}".format(BLINK_COUNTER), (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Blink detection
        if ear < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH and not EYES_CLOSED:
            BLINK_COUNTER += 1
            EYES_CLOSED = True
        elif ear >= EYE_AR_THRESH and rightEAR >= EYE_AR_THRESH:
            EYES_CLOSED = False

    cv2.imshow("Frame", frame)
    cv2.moveWindow("Frame", 450, 0)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
