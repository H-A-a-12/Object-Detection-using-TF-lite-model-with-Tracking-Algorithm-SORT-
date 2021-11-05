import os
import colorsys
import random
import cv2
import numpy as np
from keras import backend as K
import tensorflow as tf
import time
from sort  import Sort
# global track, memory_old
# memory_old={}
# track=Sort()
def Tracking(x):
        
    global track, memory_old
    
    tracks = track.update(x)
    indexids=[]
    memory = {}
    c=0
    for track in tracks:
        track_boxes = [track[0], track[1], track[2], track[3]]
        track_id = int(track[4])
        indexids.append(track_id)
        memory[track_id] = track_boxes
        c=c+1
    #memory_old = memory
    frame_memory = {key:memory[key] for key in memory if key in memory_old}
    #time.sleep(.01)

    if c%5==0:

        memory_old = memory
    #memory_old = memory
    # for i, value in enumer
    return track_id







def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def preprocess_image(image, model_image_size=(300,300)):    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, tuple(reversed(model_image_size)), interpolation=cv2.INTER_AREA)
    image = np.array(image, dtype='float32')
    image = np.expand_dims(image, 0)  # Add batch dimension.

    return image

def preprocess_image_for_tflite(image, model_image_size=300):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model_image_size, model_image_size))
    image = np.expand_dims(image, axis=0)
    image = (2.0 / 255.0) * image - 1.0
    image = image.astype('float32')

    return image

def non_max_suppression(scores, boxes, classes, max_boxes=10, min_score_thresh=0.5):
    out_boxes = []
    out_scores = []
    out_classes = []
    det=[]
    if not max_boxes:
        max_boxes = boxes.shape[0]
    for i in range(min(max_boxes, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            out_boxes.append(boxes[i])
            out_scores.append(scores[i])
            out_classes.append(classes[i])
           
    out_boxes = np.array(out_boxes)
    out_scores = np.array(out_scores)
    out_classes = np.array(out_classes)
    
    return out_scores, out_boxes, out_classes


tracker=Sort()
memory_old={}
def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    h, w, _ = image.shape
    dets=[]
    global tracker, memory_old
    
    
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        ###############################################
        # yolo
        #top, left, bottom, right = box
        ###############################################

        ###############################################
        # ssd_mobilenet
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = (xmin * w, xmax * w,
                                  ymin * h, ymax * h)
        ###############################################
        try:
            dets.append([ymin, xmin, ymax, xmax])
            print(dets)
        except:
            print('No detection')
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
                
        # colors: RGB, opencv: BGR
        cv2.rectangle(image, (left, top), (right, bottom), tuple(reversed(colors[c])), 6)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
        label_rect_left, label_rect_top = int(left - 3), int(top - 3)
        label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top - 5 - label_size[1])
        cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom), tuple(reversed(colors[c])), -1)
        cx=int((left+right)//2)
        cy=int((top+bottom)//2)
        #cv2.line(image,(label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom),(255,255,0))
        cv2.circle(image,(cx,cy),8,(255,0,0),-1)
        #cv2.circle(image,(label_rect_left,label_rect_top),8,(255,0,0),-1)
        cv2.putText(image, label, (left, int(top - 4)), font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # global track, memory_old
        if len(dets)!=0:
            dets=np.asarray(dets)
            print('***************',dets)
            #t=Tracking(dets)
            tracks = tracker.update(dets)
            indexids=[]
            memory = {}
            
            for track in tracks:
                track_boxes = [track[0], track[1], track[2], track[3]]
                track_id = int(track[4])
                indexids.append(track_id)
                memory[track_id] = track_boxes
                c=c+1
                print('ID of the Object',track_id) 
                cv2.putText(image,"ID_%d"%track_id,(cx+10,cy+10),cv2.FONT_HERSHEY_COMPLEX, .5, (255,255,255),1)

            #memory_old = memory
            frame_memory = {key:memory[key] for key in memory if key in memory_old}
            
            time.sleep(.01)

            #if c%5==0:

            memory_old = memory
            # if track_id !=0:
                
        

