from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyttsx3

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        engine.say(msg)
        engine.runAndWait()

    if alarm_status2:
        saying = True
        engine.say(msg)
        engine.runAndWait()
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Argument parser for webcam index
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

# Constants
EYE_AR_THRESH = 0.17
EYE_AR_CONSEC_FRAMES = 40
YAWN_THRESH = 20
BLINK_THRESH = 20  # Adjust this value according to your need
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
YARN_COUNTER = 0
BLINK_COUNTER = 0  # Counter for blink detection
EYES_CLOSED = False  # Flag to track if eyes are closed
TIME_FRAME = 10  # Time frame for counting yawn
start_time = time.time()
MOUTH_OPEN = False  # Flag to track if the mouth is open
# Loading the predictor and detector
print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Starting video stream
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm, args=('Wake up, sir',))
                    t.daemon = True
                    t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False

        if distance > YAWN_THRESH:
            if not MOUTH_OPEN:
                YARN_COUNTER += 1
                MOUTH_OPEN = True
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if alarm_status2 == False and saying == False:
                    alarm_status2 = True

                if YARN_COUNTER > 3:
                    t = Thread(target=alarm, args=('Take some fresh air, sir',))
                    t.daemon = True
                    t.start()
                    YARN_COUNTER = 0

        else:
            if MOUTH_OPEN:
                MOUTH_OPEN = False
                alarm_status2 = False

            if time.time() - start_time > TIME_FRAME:
                YARN_COUNTER = 0
                start_time = time.time()

        if ear < EYE_AR_THRESH:
            cv2.putText(frame, "Left Eye: Closed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif ear >= EYE_AR_THRESH:
            cv2.putText(frame, "Left Eye: Open", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate EAR for the right eye
        right_eye_ear = eye_aspect_ratio(rightEye)
        if right_eye_ear < EYE_AR_THRESH:
            cv2.putText(frame, "Right Eye: Closed", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif right_eye_ear >= EYE_AR_THRESH:
            cv2.putText(frame, "Right Eye: Open", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blink Counter: {}".format(BLINK_COUNTER), (250, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Yawn COUNTER {YARN_COUNTER}", (250, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Blink detection
        if ear < EYE_AR_THRESH and right_eye_ear < EYE_AR_THRESH and not EYES_CLOSED:
            BLINK_COUNTER += 1
            EYES_CLOSED = True  # Set flag when eyes are closed
        elif ear >= EYE_AR_THRESH and right_eye_ear >= EYE_AR_THRESH:
            EYES_CLOSED = False  # Reset flag when eyes are open

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
