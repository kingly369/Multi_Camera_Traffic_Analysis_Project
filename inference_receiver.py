from collections import deque
import sys
import math
import csv
import os
import torch
import numpy as np
import cv2
from get_lines_track3 import *  #changed getlines for track3
from sort_receiver import *
from models import *  
from utils.datasets import *
from utils.utils import *
import warnings
import time

import pycuda.autoinit
from utils.yolo_with_plugins import TrtYOLO
from utils.yolo_classes import get_cls_dict
import importlib

from PIL import Image
from torchvision import transforms
from receiveworker_receiver import *
import json

#import onnxruntime
importlib.reload(sys.modules['utils.utils'])
importlib.reload(sys.modules['utils'])

warnings.filterwarnings("ignore", category=DeprecationWarning) #ignore the sklearn 0.21 linear_assignment deprecation warning
# input: start_video_id end_video_id 
start_video = sys.argv[2]
intersection = sys.argv[1]

data_path = 'data'
datasetA_path = os.path.join(data_path, 'Dataset_A')
Track3_path = os.path.join(data_path, 'Track3/{}'.format(intersection))


#video_id_dict = get_video_id(os.path.join(data_path, 'Track3')) #change for track3
#print('video_id_dict', video_id_dict)
classes = {0:'car', 1:'truck'}
agnostic_nms = False
augment = False
cfg = os.path.join('cfg', 'yolov3.cfg')
conf_thres = 0.3
device = ''
img_size = 512
iou_thres = 0.6
output_path = 'output'
weights = os.path.join('weights', 'best.pt')

device = torch_utils.select_device(device)

trt_yolo = TrtYOLO("yolov3-288",(288, 288), 80) 
#sess = onnxruntime.InferenceSession("/home/kentngo99/project/AIC_2020_Challenge_Track-1/newencoder.onnx")
#preprocess_img = transforms.Compose([transforms.Resize(112), transforms.CenterCrop(112), transforms.ToTensor()])
#input_name = sess.get_inputs()[0].name

#sess = onnxruntime.InferenceSession("/home/jtseng/project/tensorrt_demos/yolo/yolov3-288.onnx")
#model = Darknet(cfg, img_size)
#attempt_download(weights)
#model.load_state_dict(torch.load(weights, map_location=device)['model'])
#model.to(device).eval()

#trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

def _preprocess_yolo(img, input_shape, letter_box=False):
    """Preprocess an image before TRT YOLO inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        print(input_shape[1])
        print(input_shape[0])
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img

INT_MAX = 10000

# Given three colinear points p, q, r,
# the function checks if point q lies
# on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:

    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True

    return False

# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are colinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p, q, r):
    val = (((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))

    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock

def doIntersect(p1, q1, p2, q2):

    # Find the four orientations needed for
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True

    # p1, q1 and p2 are colinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True

    # p2, q2 and p1 are colinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True

    # p2, q2 and q1 are colinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True

    return False

# Returns true if the point p lies
# inside the polygon[] with n vertices
def is_inside_polygon(points, p):

    n = len(points)

    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False

    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
    count = i = 0

    while True:
        next = (i + 1) % n

        # Check if the line segment from 'p' to
        # 'extreme' intersects with the line
        # segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i],
                        points[next],
                        p, extreme)):

            # If the point 'p' is colinear with line
            # segment 'i-next', then check if it lies
            # on segment. If it lies, return true, otherwise false
            if orientation(points[i], p,
                           points[next]) == 0:
                return onSegment(points[i], p,
                                 points[next])

            count += 1

        i = next

        if (i == 0):
            break

    # Return true if count is odd, false otherwise
    return (count % 2 == 1)

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    shape = img.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)   
    r = max(new_shape) / max(shape)
    if not scaleup:  
        r = min(r, 1.0)
    ratio = r, r  
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] 
    if auto:  
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  
    elif scaleFill:  
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  
    dw /= 2 
    dh /= 2
    if shape[::-1] != new_unpad:  
        img = cv2.resize(img, new_unpad, interpolation=interp) 
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def get_boxes(frame):
    img = letterbox(frame, new_shape=img_size)[0]
    boxestrt, confstrt, clsstrt = trt_yolo.detect(img.astype(np.float32), 0.05)
    img = img[:, :, ::-1].transpose(2, 0, 1) 
    img = np.ascontiguousarray(img)
    dettrt = scale_coords(img.shape[1:], np.array(boxestrt).astype(np.float32), frame.shape).round()
    dettrt = np.concatenate((dettrt, np.array([confstrt]).T, np.array([clsstrt]).T), axis=1)
    sorted(dettrt, key=lambda score: score[4])
    counter = 0
    while counter < len(dettrt):
        if dettrt[counter][5] == 2:
            dettrt[counter][5] = 0
            counter = counter + 1
        elif dettrt[counter][5] in [7, 6, 5]:
            dettrt[counter][5] = 1
            counter = counter + 1
        else:
            dettrt = np.delete(dettrt, counter, 0)
    return dettrt

cam_txt_id = { 1: 'c001', 2: 'c002', 3: 'c003', 4: 'c004', 5: 'c005', 6: 'c010', 7: 'c011', 8: 'c012', 9: 'c013', 10: 'c018', 11: 'c019', 12: 'c020',
        13: 'c022', 14: 'c023', 15: 'c033', 16: 'c034', 17: 'c035', 18: 'c038', 19: 'c039', 20: 'c040'}

camera_id = start_video[:-1]

for video_id in range(int(camera_id), int(camera_id)+1):
    if start_video == "15a":
        # cam 33 a
#        multi_cam_roi = [(174, 347), (296,187) , (891, 163), (1057, 346)]
        multi_cam_roi = [(115,404), (1093, 406), (891, 76), (738, 142), (590, 144), (330, 99)]
    elif start_video == "15b":
        # cam 33 b
        multi_cam_roi = [(301, 171), (253, 48), (1075, 59), (1075, 117), (632, 136), (592, 161)]
    elif start_video == "16a":
        multi_cam_roi = [(202, 158), (547, 138), (1151, 276), (1151, 590)]
    elif start_video == "17b":
        multi_cam_roi = [(154, 440), (185, 212), (504, 140), (1036, 315), (466, 556)]
    print('video_id:%s' % str(video_id))
    #video_name = video_id_dict[video_id]
    video_name = "{}{}_synced.mp4".format(cam_txt_id[video_id], start_video[-1])
    print("video name: %s" % video_name)
    cam_id = int(video_name.split('.')[0][1:].split('_')[0][:-1])
    mov_nums, lines, directions, mov_rois = get_lines(cam_id)#get_lines(cam_id) 
    roi_nums, rois = get_rois(cam_txt_id[video_id], data_path)
    counts = [0] * mov_nums
    counts_roi = [0] * roi_nums
    vs = cv2.VideoCapture(os.path.join(Track3_path, video_name)) #change to track3 path
    (W, H) = (None, None)
    writer = None
    last_frames = {}
    print("Waiting for Jackson jetson")
    channel = startupkent()
    print("Starting kent jetson")
    tracker = Sort(channel=channel, rois=multi_cam_roi)
    memory = {}
    pts = [deque(maxlen=50) for _ in range(1000000)]
    detect_flag = False
    flags = [False] * mov_nums
    indexids = [0] * mov_nums
    delays = []
    for i in range(mov_nums):
        delays.append([])
    # save output result of every video
    csv_file_processed = open(os.path.join('.', output_path, '{}.csv'.format(str(video_id) + start_video[-1])), 'w')
    csv_writer_processed = csv.writer(csv_file_processed)
    csv_writer_processed.writerow(['video_id', 'frame_id', 'movement_id', 'vehicle_class_id'])
    data = {}
    frame_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('{}_bbox.mp4'.format(cam_txt_id[video_id] + start_video[-1]), fourcc, 10.0, (1152, 648))
    start_time = time.time()
    while True:
        ret, frame = vs.read()
        if not ret:
            result = []
            result_ori = []
            for key in data:
                video_id, frame, mov, name, roi_flag = data[key][0], data[key][1], data[key][2], data[key][3], data[key][4]
                if name != name:
                    name = 1
                result_ori.append((str(video_id), str(int(frame)), str(mov + 1), str(name)))
                if roi_flag == True:
                    if int(frame) > frame_count:
                        frame = frame_count
                    result.append((str(video_id), str(int(frame)), str(mov + 1), str(name)))
                else:    
                    if len(delays[mov]) > 0:
                        frame_delay = frame + sum(delays[mov])/len(delays[mov])
                    else:
                        frame_delay = last_frames[key]
       
                    if int(frame_delay) > frame_count:
                        frame_delay = frame_count
                    result.append((str(video_id), str(int(frame_delay)), str(mov + 1), str(name)))
            csv_writer_processed.writerows(result)
            csv_file_processed.close()
            break
        frame_count += 1
        with torch.no_grad():
            bboxes = get_boxes(frame)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        boxes = []
        confidences = []
        classIDs = []
        try:
            for i, bbox in enumerate(bboxes):
                if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 60000:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
                    boxes.append([int(coor[0]), int(coor[1]), int(coor[2] - coor[0]), int(coor[3] - coor[1]), class_ind])
                    confidences.append(float(score))
                    classIDs.append(class_ind)
        except:
            print(frame_count)
        idxs = list(range(len(boxes)))
        dets = []
        record = []
        if len(idxs) > 0:
            for i in idxs:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                class_ind = bboxes[i][5]
                if classes[class_ind] == 'car' or \
                        classes[class_ind] == 'bus' or \
                        classes[class_ind] == 'truck' or \
                        classes[class_ind] == 'train':
                    if w * h < 60000: 
                        dets.append([x, y, x + w, y + h, confidences[i]])
                        center_x = int(x + 0.5 * w)
                        cneter_y = int(y + 0.5 * h)
                        record.append((center_x, cneter_y, int(class_ind)))
        new_frame = frame
        for i, boxes_element in enumerate(boxes):
            right = int(boxes_element[2] + boxes_element[0])
            bottom = int(boxes_element[3] + boxes_element[1])
            center_x = int(boxes_element[0] + 0.5 * boxes_element[2])
            center_y = int(boxes_element[1] + 0.5 * boxes_element[3])
            color = (0, 0, 0)
            if is_inside_polygon(multi_cam_roi, (center_x, center_y)):
                color = (255, 255, 0)
            else:
                if int(boxes_element[4]) == 0:
                    color = (50, 205, 50)
                else:
                    color = (225, 0, 0)
            if int(boxes_element[4]) == 0:
                new_frame = cv2.rectangle(frame, (int(boxes_element[0]), int(boxes_element[1])), (right, bottom), color, 2)
            else:
                new_frame = cv2.rectangle(frame, (int(boxes_element[0]), int(boxes_element[1])), (right, bottom), color, 2)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets, frame)
        boxes = []
        indexIDs = []
        previous = memory.copy()
        memory = {}
        for track in tracks:
            boxes.append([float(track[0]), float(track[1]), float(track[2]), float(track[3])])
            indexIDs.append(str(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
        
        for i, boxes_element in enumerate(boxes):
            new_frame = cv2.putText(new_frame, indexIDs[i],(int(boxes_element[0]), int(boxes_element[1]+15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            #new_frame = cv2.putText(new_frame, str(round(confidences[i], 2)),(int(float(boxes_element[0])), int(float(boxes_element[1])+10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        new_frame = cv2.putText(new_frame, str(frame_count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        output_video.write(new_frame)
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                if indexIDs[i] in last_frames:
                    if frame_count > last_frames[indexIDs[i]]:
                        last_frames[indexIDs[i]] = frame_count
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
                center = (int(0), int(0))
                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))      
                    for mov in range(mov_nums):
                        if intersect(p0, p1, lines[mov][0], lines[mov][1]):
                            if directions == []:
                                detect_flag = True
                                flags[mov] = True
                                indexids[mov] = indexIDs[i]
                                last_frames[indexIDs[i]] = frame_count
                                center = p0
                                break
                            if directions[mov] == 1 and x2 < x:
                                detect_flag = True
                                flags[mov] = True
                                indexids[mov] = indexIDs[i]
                                last_frames[indexIDs[i]] = frame_count
                                center = p0
                                break
                            if directions[mov] == 2 and x2 > x:
                                detect_flag = True
                                flags[mov] = True
                                indexids[mov] = indexIDs[i]
                                last_frames[indexIDs[i]] = frame_count
                                center = p0
                                break
                            if directions[mov] == 3 and y2 < y:
                                detect_flag = True
                                flags[mov] = True
                                indexids[mov] = indexIDs[i]
                                last_frames[indexIDs[i]] = frame_count
                                center = p0
                                break
                            if directions[mov] == 4 and y2 > y:
                                detect_flag = True
                                flags[mov] = True
                                indexids[mov] = indexIDs[i]
                                last_frames[indexIDs[i]] = frame_count
                                center = p0
                                break
                    for roi in range(roi_nums):
                        if intersect(p0, p1, rois[roi][0], rois[roi][1]):
                            if indexIDs[i] in data.keys():
                                delays[data[indexIDs[i]][2]].append(frame_count - data[indexIDs[i]][1])
                                data[indexIDs[i]][1] = frame_count
                                data[indexIDs[i]][4] = True
                i += 1
                if detect_flag:
                    name = '1'
                    for x in record:
                        d1 = x[0] - center[0]
                        d2 = x[1] - center[1]
                        dis = math.sqrt(d1 * d1 + d2 * d2)
                        if dis < 10:
                            name = classes[x[2]]
                            if name == 'car' or name == 'bus':
                                name = 1
                            if name == 'truck' or name == 'train':
                                name = 2
                for mov in range(mov_nums):
                    if flags[mov]:
                        counts[mov] += 1
                        roi_flag = False
                        data[indexids[mov]] = [str(video_id), frame_count, mov, name, roi_flag]
                        break
                detect_flag = False
                for mov in range(mov_nums):
                    flags[mov] = False
        frame_ready = consumeMessage(channel)
        if int(frame_ready[0]) == frame_count:
            print("On frame {}".format(frame_count))
    output_video.release()

print("--- %s running time in seconds ---" % (time.time() -start_time))
