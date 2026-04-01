from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Tuple

from image_encoder import ImageEncoder
from prompt_encoder import PromptEncoder
from mask_decoder import MaskDecoder

class GODetector:
    def __init__(self, yolo_weights_path, tracker_config_path, yolo_config=None):
        self.model = YOLO(yolo_weights_path)
        self.tracker_config_path = tracker_config_path
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.yolo_config = {
            "iou": 0.6, 
            "conf": 0.5, 
            "agnostic_nms": True, 
            "end2end": False
        } if yolo_config is None else yolo_config

    def _preprocess(self, img):
        """Convert tensor (C,H,W) → uint8 numpy (H,W,C)"""
        return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def _process_pair(self, pair):
        frame1, frame2 = pair

        results1 = self.model.track(
            frame1,
            persist=True,
            classes=[0],
            tracker=self.tracker_config_path,
            *self.yolo_config
        )

        temp = self.model.track(
            frame2,
            persist=True,
            classes=[0],
            tracker=self.tracker_config_path,
            *self.yolo_config
        )

        results2 = self.model.track(
            frame2,
            persist=True,
            classes=[0],
            tracker=self.tracker_config_path,
            *self.yolo_config
        )

        tracked_objects = {}

        if results1[0].boxes.id is not None:
            boxes1 = results1[0].boxes.xyxy
            ids1 = results1[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes1, ids1):
                tracked_objects[track_id] = {'frame1_bbox': box, 'frame2_bbox': None}

        if results2[0].boxes.id is not None:
            boxes2 = results2[0].boxes.xyxy
            ids2 = results2[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes2, ids2):
                if track_id in tracked_objects:
                    tracked_objects[track_id]['frame2_bbox'] = box
                else:
                    # Case where an ID appears only in the second frame
                    tracked_objects[track_id] = {'frame1_bbox': None, 'frame2_bbox': box}

        paired_bboxes = []
        for track_id, bboxes in tracked_objects.items():
            paired_bboxes.append({
                'id': track_id,
                'frame1_bbox': bboxes['frame1_bbox'],
                'frame2_bbox': bboxes['frame2_bbox']
            })

        return paired_bboxes, results1[0], results2[0]
    
    def forward(self, frame1_batch, frame2_batch):
        """
        Args:
            frame1_batch: Tensor (B, C, H, W)
            frame2_batch: Tensor (B, C, H, W)

        Returns:
            batch_paired_bboxes: list of length B
            results1: list of Result
            results2: list of Result
        """

        f1_list = [self._preprocess(img) for img in frame1_batch]
        f2_list = [self._preprocess(img) for img in frame2_batch]

        pairs = list(zip(f1_list, f2_list))

        outputs = []
        for pair in pairs:
            outputs.append(self._process_pair(pair))

        # Unpack
        batch_paired_bboxes = []
        results1 = []
        results2 = []

        for paired_bboxes, res1, res2 in outputs:
            batch_paired_bboxes.append(paired_bboxes)
            results1.append(res1)
            results2.append(res2)

        return batch_paired_bboxes, results1, results2

class GOSeg(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
               self,
                image_encoder: ImageEncoder,
                prompt_encoder: PromptEncoder,
                mask_decoder: MaskDecoder,
                pixel_mean: List[float] = [123.675, 116.28, 103.53],
                pixel_std: List[float] = [58.395, 57.12, 57.375],
                yolo_weights_path = "./weights/detector.pt", 
                tracker_config_path = "./botsort.yaml", 
                yolo_config = None
            ):
        super().__init__()
        self.detector = GODetector(yolo_weights_path, tracker_config_path, yolo_config)
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        
        

    def forward(self, frame1_batch, frame2_batch):

        paired_bboxes, _, _ = self.detector.forward(frame1_batch, frame2_batch)
        
  
# frame1 = cv2.imread("/content/656474110_1218013703745131_7188447242456347697_n.png")
# frame2 = cv2.imread("/content/657445712_1304225651634837_6957421237534868237_n.png")

# results1 = model.track(frame1, persist=True, classes=[0], tracker="./botsort.yaml",
#                           iou=0.6,
#                           conf=0.4,
#                           agnostic_nms=True,
#                           end2end=False
#                        )

# temp = model.track(frame2, persist=True, classes=[0], tracker="./botsort.yaml",
#                           iou=0.6,
#                           conf=0.4,
#                           agnostic_nms=True,
#                           end2end=False
#                        )

# results2 = model.track(frame2, persist=True, classes=[0], tracker="./botsort.yaml",
#                           iou=0.6,
#                           conf=0.4,
#                           agnostic_nms=True,
#                           end2end=False
#                        )

# # 5. Access tracking results
# for r in [results1[0], results2[0]]:
#     print(f"--- Tracking Results ---")
#     if r.boxes.id is not None:
#         boxes = r.boxes.xyxy.cpu().numpy()
#         ids = r.boxes.id.cpu().numpy().astype(int)
#         conf = r.boxes.conf.cpu().numpy()

#         for box, track_id, score in zip(boxes, ids, conf):
#             print(f"ID: {track_id} | Box: {box} | Conf: {score:.2f}")
#     else:

#         print("No objects tracked in this frame.")