#USEAGE: python script_with_ads.py --video "./videos/Trim 14.mp4" --frame_skipping_rate 8 --ad_time_duration 5

import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from utils import expand_bbox, draw_label, fetch_ad, get_age_range, count_people, find_dominancy

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video", type=str, required = True,
                help="path to video")

ap.add_argument("-fs", "--frame_skipping_rate", type=int,
                default = 0,
                help="enable frame skipping to for slower machines")

ap.add_argument("-c", "--confidence", type=float, 
                default=0.88,
                help="minimum probability to filter weak detections of faces")

ap.add_argument("-ad", "--ad_time_duration", type=int, 
                default=5,
                help="time duration for each ad")

args = vars(ap.parse_args())

detector = MTCNN()

img_size = 64
model = tf.keras.models.load_model("age-gender.model")
capture = cv2.VideoCapture(args["video"])

frame_skip = 0

start = time.time()
ad_path = None
    
while True:
    has_frame, frame = capture.read()
    if not has_frame:
        print('Reached the end of the video')
        break
        
    if frame_skip < args["frame_skipping_rate"]: 
        frame_skip += 1
        continue
    frame_skip = 0
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected = detector.detect_faces(img)
    faces = np.empty((len(detected), img_size, img_size, 3))
    
    detected = [d for d in detected if d['confidence'] > args["confidence"]]
    
    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    for i,face in enumerate(detected):
        bounding_box = face['box']
        xw1, yw1, xw2, yw2 = expand_bbox(frame.shape, bounding_box, margin = 0.5)

        faces[i,:,:,:] = cv2.resize(img[yw1:yw2+1, xw1:xw2+1],(img_size,img_size))#.reshape(-1,64,64,3)
        
    if len(faces) == 0: continue

    results = model.predict(faces)
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()

    prominent_gender, prominent_age = find_dominancy(count_people(results))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.putText(frame,str("{},{}".format(prominent_gender,prominent_age)), (10,20),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255))
    
    for i,face in enumerate(detected):
        label = "{}, {}".format(get_age_range(int(predicted_ages[i])),
                          "M" if predicted_genders[i][0] < 0.5 else "F")

        cv2.rectangle(frame, (face['box'][0], face['box'][1]),
                             (face['box'][0] +face['box'][2], face['box'][1]+face['box'][3]), (0, 0, 255), 2)
        draw_label(frame, (face['box'][0],face['box'][1]), label)

    if (time.time()-start) > args["ad_time_duration"]:
        start = time.time()
        ad_path = fetch_ad(prominent_gender,prominent_age)
    
    if ad_path:
        ad = cv2.imread(ad_path)
        ad_img = cv2.resize(ad, (400,600))
        frame = cv2.resize(frame, (700,600))
        frame = np.hstack((frame,ad_img))
    
    cv2.imshow('img',frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
            break
cv2.destroyAllWindows()