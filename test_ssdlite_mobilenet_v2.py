import os
import time
import cv2
import numpy as np
import tensorflow as tf
from utils.ssd_mobilenet_utils import *
from sort  import Sort


track=Sort()



# for i, value in enumerate(classes):
#     if detections['detection_scores'][i] > self.threshold:
#         if int(value)== self.class_value:
#             (y_1, x_1) = (int(detections['detection_boxes'][i][0]*width), int(detections['detection_boxes'][i][1]*height))
#             (y_2, x_2) = (int(detections['detection_boxes'][i][2]*width), int(detections['detection_boxes'][i][3]*height))
#             cv2.rectangle(self.image_np,(x_1,y_1),(x_2,y_2),(0,255,0),1)
#             self.dets.append([x_1,y_1,x_2,y_2,detections['detection_scores'][i]])
#             #cv2.putText(self.image_np,"%d"%int(value),(20,20),cv2.FONT_HERSHEY_COMPLEX, .5,(0,255,0),1)
# print('Dets',self.dets)   
# self.dets = np.asarray(self.dets)
# # self.draw_on_frame()

# if  len(self.dets)!= 0 :
# self.analytics()

def run_detection(image, interpreter):
    # Run model: start to detect
    # Sets the value of the input tensor.
    interpreter.set_tensor(input_details[0]['index'], image)
    # Invoke the interpreter.
    interpreter.invoke()

    # get results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])

    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes + 1).astype(np.int32)
#     dets.append([boxes,scores])
#     tracks = self.tracker.update(self.dets)
#     indexids=[]
#     memory = {}
#     for track in tracks:
#         track_boxes = [track[0], track[1], track[2], track[3]]
#         track_id = int(track[4])
#         indexids.append(track_id)
#         memory[track_id] = track_boxes

# frame_memory = {key:memory[key] for key in memory if key in self.memory_old}
# time.sleep(.01)



# self.memory_old = memory

    out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)

    # Print predictions info
    #print('Found {} boxes for {}'.format(len(out_boxes), 'images/dog.jpg'))
            
    return out_scores, out_boxes, out_classes

def image_object_detection(interpreter, colors):
    image = cv2.imread('images/dog.jpg')
    image_data = preprocess_image_for_tflite(image, model_image_size=300)
    out_scores, out_boxes, out_classes = run_detection(image_data, interpreter)

    # Draw bounding boxes on the image file
    result = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    cv2.imwrite(os.path.join("out", "ssdlite_mobilenet_v2_dog.jpg"), result, [cv2.IMWRITE_JPEG_QUALITY, 90])

def real_time_object_detection(interpreter, colors):
    camera = cv2.VideoCapture(0)
    counter=0
    while camera.isOpened():
        start = time.time()
        ret, frame = camera.read()
        counter=counter+1 
        if counter%2==0:
            if ret:
                image_data = preprocess_image_for_tflite(frame, model_image_size=300)
                out_scores, out_boxes, out_classes = run_detection(image_data, interpreter)
                # Draw bounding boxes on the image file
                result = draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors)
                end = time.time()

                # fps
                t = end - start
                fps  = "Fps: {:.2f}".format(1 / t)
                cv2.putText(result, fps, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
                cv2.imshow("Object detection - ssdlite_mobilenet_v2", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="model_data/ssdlite_mobilenet_v2.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # label
    class_names = read_classes('model_data/coco_classes.txt')
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
            
    #image_object_detection(interpreter, colors)
    real_time_object_detection(interpreter, colors)
