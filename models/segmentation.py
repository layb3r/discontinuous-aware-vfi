from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Tuple

from .image_encoder import TinyViT
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder

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
            **self.yolo_config
        )

        temp = self.model.track(
            frame2,
            persist=True,
            classes=[0],
            tracker=self.tracker_config_path,
            **self.yolo_config
        )

        results2 = self.model.track(
            frame2,
            persist=True,
            classes=[0],
            tracker=self.tracker_config_path,
            **self.yolo_config
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
        for track_id, bboxes in sorted(tracked_objects.items(), key=lambda x: x[0]):
            paired_bboxes.append({
                'id': track_id,
                'frame1_bbox': bboxes['frame1_bbox'],
                'frame2_bbox': bboxes['frame2_bbox']
            })

        return paired_bboxes, results1[0], results2[0]

    def _to_batched_tensor(self, batch_paired_bboxes: List[List[Dict[str, Any]]], device: torch.device):
        """
        Convert variable-length paired detection output to padded tensors.

        Returns:
            paired_boxes: (B, N, 2, 4) float32
            paired_valid: (B, N, 2) bool
            paired_ids: (B, N) long
        """
        batch_size = len(batch_paired_bboxes)
        max_instances = max((len(pairs) for pairs in batch_paired_bboxes), default=0)

        paired_boxes = torch.zeros((batch_size, max_instances, 2, 4), dtype=torch.float32, device=device)
        paired_valid = torch.zeros((batch_size, max_instances, 2), dtype=torch.bool, device=device)
        paired_ids = torch.full((batch_size, max_instances), -1, dtype=torch.long, device=device)

        for b, pair_list in enumerate(batch_paired_bboxes):
            for i, item in enumerate(pair_list):
                paired_ids[b, i] = int(item['id'])

                if item['frame1_bbox'] is not None:
                    paired_boxes[b, i, 0] = item['frame1_bbox'].detach().to(device=device, dtype=torch.float32)
                    paired_valid[b, i, 0] = True

                if item['frame2_bbox'] is not None:
                    paired_boxes[b, i, 1] = item['frame2_bbox'].detach().to(device=device, dtype=torch.float32)
                    paired_valid[b, i, 1] = True

        return paired_boxes, paired_valid, paired_ids
    
    @torch.no_grad()
    def forward(self, frame1_batch, frame2_batch):
        """
        Args:
            frame1_batch: Tensor (B, C, H, W)
            frame2_batch: Tensor (B, C, H, W)

        Returns:
            paired_boxes: Tensor (B, N, 2, 4)
            paired_valid: Tensor (B, N, 2)
            paired_ids: Tensor (B, N)
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

        paired_boxes, paired_valid, paired_ids = self._to_batched_tensor(
            batch_paired_bboxes,
            device=frame1_batch.device,
        )

        return paired_boxes, paired_valid, paired_ids, results1, results2

class GOSeg(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
               self,
                image_encoder: TinyViT,
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

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def _segment_with_boxes(
        self,
        frame: torch.Tensor,
        boxes: torch.Tensor,
        multimask_output: bool,
    ) -> Dict[str, torch.Tensor]:
        """Run box-prompted segmentation for one image."""
        num_masks = self.mask_decoder.num_multimask_outputs if multimask_output else 1

        if boxes.numel() == 0:
            h, w = frame.shape[-2:]
            return {
                "masks": torch.zeros((0, num_masks, h, w), dtype=torch.bool, device=frame.device),
                "iou_predictions": torch.zeros((0, num_masks), dtype=torch.float32, device=frame.device),
                "low_res_logits": torch.zeros((0, num_masks, 256, 256), dtype=torch.float32, device=frame.device),
            }

        input_image = self.preprocess(frame).unsqueeze(0)
        image_embeddings = self.image_encoder(input_image)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        masks = self.postprocess_masks(
            low_res_masks,
            input_size=frame.shape[-2:],
            original_size=frame.shape[-2:],
        )
        masks = masks > self.mask_threshold

        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }

    @torch.no_grad()
    def forward(self, frame1_batch, frame2_batch, multimask_output: bool):
        paired_boxes, paired_valid, paired_ids, _, _ = self.detector.forward(frame1_batch, frame2_batch)

        frame1_outputs = []
        frame2_outputs = []

        for b in range(frame1_batch.shape[0]):
            frame1_boxes = paired_boxes[b, paired_valid[b, :, 0], 0, :]
            frame2_boxes = paired_boxes[b, paired_valid[b, :, 1], 1, :]

            frame1_outputs.append(
                self._segment_with_boxes(
                    frame=frame1_batch[b],
                    boxes=frame1_boxes,
                    multimask_output=multimask_output,
                )
            )
            frame2_outputs.append(
                self._segment_with_boxes(
                    frame=frame2_batch[b],
                    boxes=frame2_boxes,
                    multimask_output=multimask_output,
                )
            )

        return {
            "paired_boxes": paired_boxes,
            "paired_valid": paired_valid,
            "paired_ids": paired_ids,
            "frame1_outputs": frame1_outputs,
            "frame2_outputs": frame2_outputs,
        }

  
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