import os
import cv2
import time
import argparse
import torch
import numpy as np
import json

from collections import deque
from predict import InferYOLOv3
from utils.utils import xyxy2xywh
from deep_sort import DeepSort
from utils.utils_sort import COLORS_10, draw_bboxes


from matplotlib import pyplot as plt
from build_utils import img_utils
from build_utils import torch_utils
from build_utils import utils
from models_spp import Darknet
from draw_box_utils import draw_box

def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    for i in range(3):
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
        
    return sceneRadiance


def xyxy2tlwh(x):
    '''
    (top left x, top left y,width, height)
    '''
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


class Detector(object):
    def __init__(self, args):
        self.args = args
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.vdo = cv2.VideoCapture()
        self.yolo3 = InferYOLOv3(args.yolo_cfg,
                                 args.img_size,
                                 args.yolo_weights,
                                 args.data_cfg,
                                 device,
                                 conf_thres=args.conf_thresh,
                                 nms_thres=args.nms_thresh)
        self.deepsort = DeepSort(args.deepsort_checkpoint)

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20,
                                          (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self, outfile=None):
        frame_cnt = -1

        if outfile is not None:
            f = open(outfile, 'w')
        
        print("begin....")

        while self.vdo.grab():
            frame_cnt += 1

            if frame_cnt % 3 == 0:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = ori_im

            t1_begin = time.time()
            bbox_xxyy, cls_conf, cls_ids = self.yolo3.predict(im)
            t1_end = time.time()

            t2_begin = time.time()
            if bbox_xxyy is not None:
                # select class
                # mask = cls_ids == 0
                # bbox_xxyy = bbox_xxyy[mask]

                # bbox_xxyy[:, 3:] *= 1.2
                # cls_conf = cls_conf[mask]

                bbox_xcycwh = xyxy2xywh(bbox_xxyy)
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    # 画框
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)

                    # frame, id, tlwh(%.2f),1,-1,-1,-1
                    if outfile is not None:
                        box_xywh = xyxy2tlwh(bbox_xyxy)
                        for i in range(len(box_xywh)):
                            write_line = "%d,%d,%d,%d,%d,%d,1,-1,-1,-1\n" % (
                                frame_cnt +
                                1, outputs[i, -1], int(box_xywh[i]
                                                       [0]), int(box_xywh[i][1]),
                                int(box_xywh[i][2]), int(box_xywh[i][3]))
                            f.write(write_line)

            t2_end = time.time()

            end = time.time()
            print(
                "frame:%d|det:%.4f|sort:%.4f|total:%.4f|det p:%.2f%%|fps:%.2f"
                % (frame_cnt, (t1_end - t1_begin), (t2_end - t2_begin),
                   (end - start), ((t1_end - t1_begin) * 100 /
                                   ((end - start))), (1 / (end - start))))
            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(ori_im)

        if outfile is not None:
            f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--yolo_cfg",
                        type=str,
                        default="../YOLOv3-complete-pruning-master/cfg/dense-v3-tiny-spp.cfg"
                        ) 
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="./weights/yolov3spp-59.pt"
    )
    parser.add_argument("--conf_thresh", type=float, default=0.5)  # ori 0.5
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    parser.add_argument("--deepsort_checkpoint",
                        type=str,
                        default="deep_sort/deep/checkpoint/mycpkt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display",
                        dest="display",
                        action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--data_cfg", type=str, default="data/voc_small.data")
    parser.add_argument("--img_size", type=int, default=512, help="img size")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_file = "./data/videosample/predicts.txt"
    with Detector(args) as det:
        det.detect(output_file)

    os.system("ffmpeg -y -i demo.avi -r 10 -b:a 32k %s_output.mp4" %
              (os.path.basename(args.VIDEO_PATH).split('.')[0]))
