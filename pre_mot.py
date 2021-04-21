import argparse
import os
import random
import time
from os.path import join
import json

import cv2
import numpy as np
import torch
from draw_box_utils import draw_box

from deep_sort import DeepSort
from predict import InferYOLOv3
from utils.utils import xyxy2xywh
from utils.utils_sort import COLORS_10, draw_bboxes


def xyxy2tlwh(x):
    '''
    (transfer to upper left x, upper left y, width, height)
    '''
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

# apply CLAHE 
def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    for i in range(3):
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance

# display the count and ratio on the screen
def displayNephropsCount(frame, nephrops_count):
    cv2.putText(frame,'Nephrops: ' + str(nephrops_count),(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2,cv2.FONT_HERSHEY_COMPLEX_SMALL)
def displayFlatfishCount(frame, flatfish_count):
    cv2.putText(frame,'Flat fish: ' + str(flatfish_count),(20, 80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2,cv2.FONT_HERSHEY_COMPLEX_SMALL)   
def displayRoundfishCount(frame, roundfish_count):
    cv2.putText(frame,'Round fish: ' + str(roundfish_count),(20, 120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2,cv2.FONT_HERSHEY_COMPLEX_SMALL)
def displayOtherfishCount(frame, otherfish_count):
    cv2.putText(frame,'Other fish: ' + str(otherfish_count),(20, 160),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2,cv2.FONT_HERSHEY_COMPLEX_SMALL)
def displayCatchRatio(frame, catch_ratio):
    cv2.putText(frame,'Catch Ratio: ' + str(catch_ratio), (20, 200),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2,cv2.FONT_HERSHEY_COMPLEX_SMALL)
def displayByCatchRatio(frame, bycatch_ratio):
    cv2.putText(frame,'By-Catch Ratio: ' + str(bycatch_ratio), (20, 240),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2,cv2.FONT_HERSHEY_COMPLEX_SMALL)

## Detector and counter based on the deep sort algorithm
'''
parameter description
- note that all the info can be modified in the following args settings

cfg: yolov3-spp cfg file path
weights: yolov3-spp weights path
video_path: video to be processed path
deep_checkpoint: feature extractor network(here I used Restnet50) weights path
output_file: a path, recording coordinates info in a txt file
img_size: define img size
display: define if it show the figure online
max_dist: define max_dist for deepsort
display_width: define display width of the window
display_height: define display height of the window
save_path: the saving path of new generated video
json_path: json file containing class info used for drawing bbx
device: define the device cuda or cpu machine
'''
class DeepSortDetector(object):
    def __init__(
            self,
            cfg,
            weights,
            video_path,
            deep_checkpoint="deep_sort/deep/checkpoint/resnet50_last.pt",
            output_file=None,
            img_size=512,
            display=True,
            max_dist=0.2,
            display_width=800,
            display_height=600,
            save_path=None,
            json_path='./data/pascal_voc_classes.json'):
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        # init opencv video capturer    
        self.vidCap = cv2.VideoCapture()
        # init a detector
        self.yolov3 = InferYOLOv3(cfg, img_size, weights, device,
                                  json_path)
        # init a deepsort tracker
        self.deepsort = DeepSort(deep_checkpoint, max_dist)
        # settings
        self.display = display
        self.video_path = video_path
        self.output_file = output_file
        self.save_path = save_path

        if self.display:
            cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Test", display_width, display_height)

    # define a video writter named self.output
    def __enter__(self):
        assert os.path.isfile(self.video_path), "Error: path error"
        self.vidCap.open(self.video_path)
        self.im_width = int(self.vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #self.im_width = 1280
        #self.im_height = 720

        if self.save_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.save_path, fourcc, 15.0,(self.im_width, self.im_height))
        assert self.vidCap.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    # this is the key function to detect and count fishes
    def detect(self):
        json_path = './data/pascal_voc_classes.json'
        json_file = open(json_path, 'r')
        class_dict = json.load(json_file)
        category_index = {v: k for k, v in class_dict.items()}
        # All these classes will be counted as 'catch'
        list_of_catch = ["nephrops", "flat_fish", "round_fish"]
        # these classes will be counted as 'by-catch'
        list_of_bycatch = ["other"]
        LABELS = ['flat_fish', 'round_fish', 'nephrops', 'other']
        # to store the object infomation key:id value: class
        all_obj_info = {}

        frame_no = -1
        num_frames, nephrops_count, flatfish_count, roundfish_count, other_count = 0, 0, 0, 0, 0
        catch_ratio, bycatch_ratio = 0, 0
        # skip_no = 2

        if self.output_file:
            f = open(output_file, "w")

        while self.vidCap.grab():
            frame_no += 1

            # skip frames every n frames
            # if frame_no % skip_no != 0:
            #     continue

            # start time
            total_begin = time.time()

            _, img = self.vidCap.retrieve()
            #img = img[:, :1280]

            # yolov3
            yolo_begin = time.time()
            # get the detections: bbx coordinates, confidences, classes
            bbox_xyxy_ori, cls_conf, cls_ids = self.yolov3.predict(img)
            print(cls_ids)
            
            # [x1,y1,x2,y2]
            yolo_end = time.time()

            # deepsort
            ds_begin = time.time()
            if bbox_xyxy_ori is not None:
                # transfer the coorinates
                bbox_cxcywh = xyxy2xywh(bbox_xyxy_ori)
                # use the tracker to update
                outputs = self.deepsort.update(bbox_cxcywh, cls_conf, cls_ids, img)

                if len(outputs) > 0:
                    # [x1,y1,x2,y2] id class
                    # now we can fetch the bbx info, ids and classes
                    bbox_xyxy = outputs[:, :4]
                    ids = outputs[:, -2]
                    object_class = outputs[:, -1]
                    print(ids)
                    print(object_class)

                    ## obj_id and class alignment has some problems
                    #  it is hard to be very acurate
                    # need to make it better 
                    # for i in range(len(ids)):
                    #     if ids[i] not in all_obj_info:
                    #         if len(cls_ids) == len(ids) - 1:
                    #             all_obj_info[ids[i]] = cls_ids[i-1]
                    #         elif len(cls_ids) == len(ids) - 2:
                    #             all_obj_info[ids[i]] = cls_ids[i-2]
                    #         elif len(cls_ids) == len(ids) - 3:
                    #             all_obj_info[ids[i]] = cls_ids[i-3]
                    #         elif len(cls_ids) == len(ids) - 4:
                    #             all_obj_info[ids[i]] = cls_ids[i-4]
                    #         elif len(cls_ids) == len(ids) - 5:
                    #             all_obj_info[ids[i]] = cls_ids[i-5]
                    #         elif len(cls_ids) == len(ids) - 6:
                    #             all_obj_info[ids[i]] = cls_ids[i-6]
                    #         elif len(cls_ids) == len(ids) - 7:
                    #             all_obj_info[ids[i]] = cls_ids[i-7]
                    #         elif len(cls_ids) == len(ids) - 8:
                    #             all_obj_info[ids[i]] = cls_ids[i-8]
                    #         elif len(cls_ids) == len(ids) - 9:
                    #             all_obj_info[ids[i]] = cls_ids[i-9]
                    #         elif len(cls_ids) == len(ids) - 10:
                    #             all_obj_info[ids[i]] = cls_ids[i-10]
                    #         else:
                    #             all_obj_info[ids[i]] = cls_ids[i]
                    for i in range(len(ids)):
                        if ids[i] not in all_obj_info:
                            all_obj_info[ids[i]] = object_class[i]
                        else:
                            continue
                    print(all_obj_info)

                    # draw the bbx
                    img = draw_box(img, bbox_xyxy_ori, cls_ids, cls_conf, category_index)
                    #img = draw_bboxes(img, bbox_xyxy, ids)

                    # frame,id,tlwh,1,-1,-1,-1
                    # record the info
                    if self.output_file:
                        bbox_tlwh = xyxy2xywh(bbox_xyxy)
                        for i in range(len(bbox_tlwh)):
                            write_line = "%d,%d,%d,%d,%d,%d,1,-1,-1,-1\n" % (
                                frame_no + 1, outputs[i, -1],
                                int(bbox_tlwh[i][0]), int(bbox_tlwh[i][1]),
                                int(bbox_tlwh[i][2]), int(bbox_tlwh[i][3]))
                            f.write(write_line)
            ds_end = time.time()

            total_end = time.time()

            # count the current number of each category
            cur_categories = list(all_obj_info.values())
            flatfish_count = cur_categories.count(1) 
            roundfish_count = cur_categories.count(2)  
            nephrops_count = cur_categories.count(3)
            other_count = cur_categories.count(4) 
            # start from frame 3 
            if frame_no >= 3: 
                catch_ratio = round((flatfish_count + roundfish_count + nephrops_count) / (flatfish_count + roundfish_count + nephrops_count + other_count), 2)
                bycatch_ratio = round(other_count / (flatfish_count + roundfish_count + nephrops_count + other_count), 2)
            else:
                catch_ratio = None
                bycatch_ratio = None

            # print info to the console
            if frame_no is not None:
                print("frame:%04d|det:%.4f|deep sort:%.4f|total:%.4f|det p:%.2f%%|fps:%.2f" % (frame_no,
                                                                                               (yolo_end - yolo_begin),
                                                                                               (ds_end - ds_begin),
                                                                                               (total_end - total_begin),
                                                                                               ((yolo_end - yolo_begin) * 100 / (
                                                                                                   total_end - total_begin)),
                                                                                               (1 / (total_end - total_begin))))
            # display all the count info on the screen
            if self.display == True:
                img = np.uint8(img)
                displayNephropsCount(img, nephrops_count)
                displayFlatfishCount(img, flatfish_count)
                displayRoundfishCount(img, roundfish_count)
                displayOtherfishCount(img, other_count)
                displayCatchRatio(img, catch_ratio)
                displayByCatchRatio(img, bycatch_ratio)
                cv2.putText(img,'FPS {:.1f}'.format(1 / (total_end - total_begin)),(20, 280),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2,cv2.FONT_HERSHEY_COMPLEX_SMALL)   
                cv2.imshow("Test", img)
                cv2.waitKey(1)

                # press Q to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # determine if output the new video
            if self.save_path:
                self.output.write(img)

        if self.output_file:
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('parser')
    parser.add_argument("--video_root", type=str, default="./imgAndVideos/video/")
    parser.add_argument("--cfg", type=str, default="cfg/my_yolov3.cfg")
    parser.add_argument("--weights",
                        type=str,
                        default="./weights/yolov3spp-59.pt")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument(
        "--deep_checkpoint",
        type=str,
        default="deep_sort/deep/checkpoint/resnet50_last.pt")

    # hyperparameter
    parser.add_argument("--max_dist", type=float, default=0.4)

    # presentation
    parser.add_argument("--display", dest="display", action="store_true", default=True)
    parser.add_argument("--display_width", type=int, default=1280)
    parser.add_argument("--display_height", type=int, default=720)

    args = parser.parse_args()

    # all the necessary path
    video_path = args.video_root + 'Nep_plus_cod_Haul_394_204.mp4'
    output_file = "./data/videoresult/" + 'Nep_plus_cod_Haul_394_204_new.txt'
    save_path = "./output/" + "Nep_plus_cod_Haul_394_204_new.avi"
    json_path = './data/pascal_voc_classes.json'

    # init an instance to use 
    with DeepSortDetector(args.cfg, args.weights, video_path,
                            args.deep_checkpoint, output_file,
                            args.img_size, args.display, 
                            args.max_dist,
                            args.display_width, args.display_height,
                            save_path, 
                            json_path) as det:
        det.detect()

        avi_name = os.path.basename(video_path).split(".")[0]
        # os.system("ffmpeg -y -i ./output/%s.avi -r 10 -b:a 32k ./output/%s.mp4" %
        #           (avi_name, avi_name))
