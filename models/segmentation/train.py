import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .build_seg import build_goseg
except ImportError:
    from models.segmentation.build_seg import build_goseg


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class PairedSegDataset(Dataset):
    """Dataset layout:

        root/
            instance_001/
                I0.png
                I2.png
                overlays_masks/
                    000_M0.png
                    000_M2.png
                    001_M0.png
                    001_M2.png
                bboxes.json (optional)
            instance_002/
                ...

        If bboxes.json is missing, bounding boxes are derived from overlay masks.
    """

    def __init__(self, root: str, image_size: int = 1024) -> None:
        self.root = Path(root)
        self.image_size = image_size

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.samples: List[Tuple[Path, Path, Path]] = []
        for instance_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            i0_path = instance_dir / "I0.png"
            i2_path = instance_dir / "I2.png"
            overlays_dir = instance_dir / "overlays_masks"

            if not i0_path.exists() or not i2_path.exists() or not overlays_dir.exists():
                continue

            self.samples.append((instance_dir, i0_path, i2_path))

        if not self.samples:
            raise RuntimeError(
                "No valid instances found. Each instance must contain I0.png, I2.png, and overlays_masks/."
            )

    def _load_rgb(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
        arr = np.asarray(mask, dtype=np.float32)
        return torch.from_numpy((arr > 0).astype(np.float32)).contiguous()

    def _mask_to_bbox(self, mask: torch.Tensor):
        ys, xs = torch.where(mask > 0)
        if ys.numel() == 0:
            return None
        x1 = float(xs.min().item())
        y1 = float(ys.min().item())
        x2 = float(xs.max().item())
        y2 = float(ys.max().item())
        return [x1, y1, x2, y2]

    def _load_bbox_annotations(self, instance_dir: Path):
        candidates = [instance_dir / "bboxes.json", instance_dir / "boxes.json"]
        ann_path = next((p for p in candidates if p.exists()), None)
        if ann_path is None:
            return {}

        with ann_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        parsed = {}
        if isinstance(payload, dict) and "pairs" in payload and isinstance(payload["pairs"], list):
            for idx, item in enumerate(payload["pairs"]):
                if not isinstance(item, dict):
                    continue
                key = str(item.get("key", item.get("id", idx))).zfill(3)
                parsed[key] = {
                    "id": int(item.get("id", idx)),
                    "frame1_bbox": item.get("frame1_bbox", item.get("frame1")),
                    "frame2_bbox": item.get("frame2_bbox", item.get("frame2")),
                }
            return parsed

        if isinstance(payload, dict):
            for key, value in payload.items():
                if not isinstance(value, dict):
                    continue
                parsed[str(key).zfill(3)] = {
                    "id": int(value.get("id", 0)),
                    "frame1_bbox": value.get("frame1_bbox", value.get("frame1")),
                    "frame2_bbox": value.get("frame2_bbox", value.get("frame2")),
                }
        return parsed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        instance_dir, frame1_path, frame2_path = self.samples[idx]
        overlays_dir = instance_dir / "overlays_masks"

        mask0_files = sorted(overlays_dir.glob("*_M0.png"))
        if not mask0_files:
            raise RuntimeError(f"No *_M0.png files found in {overlays_dir}")

        bbox_ann = self._load_bbox_annotations(instance_dir)

        union_mask1 = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
        union_mask2 = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
        pairs = []

        for pair_idx, m0_path in enumerate(mask0_files):
            key = m0_path.stem.replace("_M0", "")
            m2_path = overlays_dir / f"{key}_M2.png"
            if not m2_path.exists():
                continue

            m0 = self._load_mask(m0_path)
            m2 = self._load_mask(m2_path)

            union_mask1 = torch.maximum(union_mask1, m0)
            union_mask2 = torch.maximum(union_mask2, m2)

            ann = bbox_ann.get(key, {})

            frame1_bbox = ann.get("frame1_bbox") if isinstance(ann, dict) else None
            frame2_bbox = ann.get("frame2_bbox") if isinstance(ann, dict) else None

            if frame1_bbox is None:
                frame1_bbox = self._mask_to_bbox(m0)
            if frame2_bbox is None:
                frame2_bbox = self._mask_to_bbox(m2)

            pair_id = int(ann.get("id", int(key) if key.isdigit() else pair_idx)) if isinstance(ann, dict) else pair_idx
            pairs.append(
                {
                    "id": pair_id,
                    "frame1_bbox": frame1_bbox,
                    "frame2_bbox": frame2_bbox,
                }
            )

        if not pairs:
            raise RuntimeError(f"No valid mask pairs found in {overlays_dir}")

        return {
            "frame1": self._load_rgb(frame1_path),
            "frame2": self._load_rgb(frame2_path),
            "mask1": union_mask1,
            "mask2": union_mask2,
            "pairs": pairs,
            "id": instance_dir.name,
        }


def collate_batch(batch):
    batch_size = len(batch)
    max_instances = max((len(x["pairs"]) for x in batch), default=0)

    paired_boxes = torch.zeros((batch_size, max_instances, 2, 4), dtype=torch.float32)
    paired_valid = torch.zeros((batch_size, max_instances, 2), dtype=torch.bool)
    paired_ids = torch.full((batch_size, max_instances), -1, dtype=torch.long)

    for b, sample in enumerate(batch):
        for i, pair in enumerate(sample["pairs"]):
            paired_ids[b, i] = int(pair["id"])
            if pair["frame1_bbox"] is not None:
                paired_boxes[b, i, 0] = torch.tensor(pair["frame1_bbox"], dtype=torch.float32)
                paired_valid[b, i, 0] = True
            if pair["frame2_bbox"] is not None:
                paired_boxes[b, i, 1] = torch.tensor(pair["frame2_bbox"], dtype=torch.float32)
                paired_valid[b, i, 1] = True

    return {
        "frame1": torch.stack([x["frame1"] for x in batch], dim=0),
        "frame2": torch.stack([x["frame2"] for x in batch], dim=0),
        "mask1": torch.stack([x["mask1"] for x in batch], dim=0),
        "mask2": torch.stack([x["mask2"] for x in batch], dim=0),
        "paired_boxes": paired_boxes,
        "paired_valid": paired_valid,
        "paired_ids": paired_ids,
        "id": [x["id"] for x in batch],
    }


def aggregate_logits_from_pairs(per_pair_outputs, frame_key: str, target_hw: Tuple[int, int], device: torch.device):
    per_object_logits = []
    for pair_item in per_pair_outputs:
        frame_item = pair_item.get(frame_key)
        if frame_item is None:
            continue

        logits = frame_item["low_res_logits"]
        if logits.ndim == 3:
            logits = logits[0]
        elif logits.ndim != 2:
            raise RuntimeError(f"Unexpected low_res_logits shape: {tuple(logits.shape)}")

        per_object_logits.append(logits)

    if not per_object_logits:
        # Strongly negative logits -> near-zero probability when no object is predicted.
        return torch.full(target_hw, -10.0, device=device)

    stacked = torch.stack(per_object_logits, dim=0)
    return torch.amax(stacked, dim=0)


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    bce = torch.nn.BCEWithLogitsLoss()
    running_loss = 0.0

    for batch in loader:
        frame1 = batch["frame1"].to(device)
        frame2 = batch["frame2"].to(device)
        mask1 = batch["mask1"].to(device)
        mask2 = batch["mask2"].to(device)
        paired_boxes = batch["paired_boxes"].to(device)
        paired_valid = batch["paired_valid"].to(device)
        paired_ids = batch["paired_ids"].to(device)

        outputs = model(
            frame1,
            frame2,
            multimask_output=False,
            paired_boxes=paired_boxes,
            paired_valid=paired_valid,
            paired_ids=paired_ids,
        )
        paired_outputs = outputs["paired_outputs"]

        pred1_list = []
        pred2_list = []
        for per_pair in paired_outputs:
            pred1_list.append(aggregate_logits_from_pairs(per_pair, "frame1", (256, 256), device))
            pred2_list.append(aggregate_logits_from_pairs(per_pair, "frame2", (256, 256), device))

        pred1 = torch.stack(pred1_list, dim=0)
        pred2 = torch.stack(pred2_list, dim=0)

        target1 = F.interpolate(mask1.unsqueeze(1), size=pred1.shape[-2:], mode="nearest").squeeze(1)
        target2 = F.interpolate(mask2.unsqueeze(1), size=pred2.shape[-2:], mode="nearest").squeeze(1)

        loss1 = bce(pred1, target1) + dice_loss_with_logits(pred1, target1)
        loss2 = bce(pred2, target2) + dice_loss_with_logits(pred2, target2)
        loss = 0.5 * (loss1 + loss2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    bce = torch.nn.BCEWithLogitsLoss()
    running_loss = 0.0

    for batch in loader:
        frame1 = batch["frame1"].to(device)
        frame2 = batch["frame2"].to(device)
        mask1 = batch["mask1"].to(device)
        mask2 = batch["mask2"].to(device)
        paired_boxes = batch["paired_boxes"].to(device)
        paired_valid = batch["paired_valid"].to(device)
        paired_ids = batch["paired_ids"].to(device)

        outputs = model(
            frame1,
            frame2,
            multimask_output=False,
            paired_boxes=paired_boxes,
            paired_valid=paired_valid,
            paired_ids=paired_ids,
        )
        paired_outputs = outputs["paired_outputs"]

        pred1_list = []
        pred2_list = []
        for per_pair in paired_outputs:
            pred1_list.append(aggregate_logits_from_pairs(per_pair, "frame1", (256, 256), device))
            pred2_list.append(aggregate_logits_from_pairs(per_pair, "frame2", (256, 256), device))

        pred1 = torch.stack(pred1_list, dim=0)
        pred2 = torch.stack(pred2_list, dim=0)

        target1 = F.interpolate(mask1.unsqueeze(1), size=pred1.shape[-2:], mode="nearest").squeeze(1)
        target2 = F.interpolate(mask2.unsqueeze(1), size=pred2.shape[-2:], mode="nearest").squeeze(1)

        loss1 = bce(pred1, target1) + dice_loss_with_logits(pred1, target1)
        loss2 = bce(pred2, target2) + dice_loss_with_logits(pred2, target2)
        running_loss += 0.5 * (loss1 + loss2).item()

    return running_loss / max(len(loader), 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GOSeg with paired frames and masks")
    parser.add_argument("--train-root", type=str, required=True, help="Dataset root for training split")
    parser.add_argument("--val-root", type=str, default=None, help="Dataset root for validation split")
    parser.add_argument("--yolo-weights", type=str, default=None, help="Optional YOLO weights (not needed for bbox-supervised training)")
    parser.add_argument("--tracker-config", type=str, default="models/segmentation/botsort.yaml")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Optional initial checkpoint path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairedSegDataset(args.train_root, image_size=args.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    val_loader = None
    if args.val_root:
        val_ds = PairedSegDataset(args.val_root, image_size=args.image_size)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_batch,
        )

    model = build_goseg(
        checkpoint=args.init_checkpoint,
        yolo_weights_path=args.yolo_weights,
        tracker_config_path=args.tracker_config,
        use_detector=False,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        val_loss = None
        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, device)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"epoch": epoch, "model": model.state_dict()}, save_dir / "best.pt")

        torch.save({"epoch": epoch, "model": model.state_dict()}, save_dir / f"epoch_{epoch:03d}.pt")

        if val_loss is None:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")


if __name__ == "__main__":
    main()
