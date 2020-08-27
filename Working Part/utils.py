import os
import cv2
import random
import numpy as np
import pandas as pd
from collections import Counter


def fetch_ad(prominent_gender,prominent_age):
    ads = []
    path = "D:\\AGE_DATASET\\Ads"
    ad_path = os.path.join(path,prominent_gender,str(prominent_age)[1:-1].replace(" ",""))
    for ad in os.listdir(ad_path):
        ads.append(ad)

    rand_ad = os.path.join(ad_path,ads[random.randint(0,len(ads)-1)])
    return rand_ad


def expand_bbox(frame_size, bbox, margin = .5):
    x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x2, y2 = bbox[0] + bbox[2] , bbox[1] + bbox[3]
    xw1 = max(int(x1 - margin * w), 0)
    yw1 = max(int(y1 - margin * h), 0)
    xw2 = min(int(x2 + margin * w), frame_size[1] - 1)
    yw2 = min(int(y2 + margin * h), frame_size[0] - 1)
    return xw1, yw1, xw2, yw2

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def get_age_range(age):
    age_groups = [(0,3), (4,7) ,(8,14), (15,25), (26,34), (35,48), (48,60), (60,100)] 
#    age_groups = [(0,2),(4,6),(8,12),(15,20),(25,32),(38,43),(48,53),(60,100)] 
    for i in age_groups:
        if(age in range (i[0],i[1]+1)):
            return age_groups[age_groups.index(i)]

def count_people(results):
    people = []
    
    ages_list = np.arange(0, 101).reshape(101, 1)

    for gender,age in zip(results[0],results[1]):
        try:
            pred_gender = "M" if gender[0] < 0.5 else "F"
            predicted_ages = int(age.dot(ages_list).flatten()[0])

            people.append([pred_gender, predicted_ages])
        except:
            continue
    return people


def find_dominancy(people):
    people = [(gender,get_age_range(age)) for gender,age in people]
    freq = Counter(people).most_common()
    prominent_gender, prominent_age = freq[0][0][0], freq[0][0][1]
    return prominent_gender, prominent_age

def make_image_df(path = "D:\\AGE_DATASET\\Ads"):
    df = pd.DataFrame(columns = ["Gender", "Age Group","Ad Path"])

    main_dir = "D:\\AGE_DATASET\\Ads"
    for gender in os.listdir(main_dir):
        for age_group in os.listdir(os.path.join(main_dir,gender)):
            folder_path = os.path.join(main_dir,gender,age_group)
            for ad_name in os.listdir(folder_path):
                df = df.append({'Gender': gender, 'Age Group': age_group, 'Ad Path': os.path.join(main_dir,folder,folder_path,ad_name)}, ignore_index=True)
    
    return df
    