"""
YOLOv8-seg builder (seg task) - YOLOv8-n (detectors/yolov8, unchanged) + the Ultralytics Segment head.

Reference: Ultralytics (2023) ultralytics/cfg/models/v8/yolov8-seg.yaml (head `Segment, [nc, 32, 256]`), the same
Segment / Proto / v8SegmentationLoss as YOLO11-seg; implementation shared with detectors/yolo11_seg (Proto,
mask loss, decode_masks), see that file for the details and the [Unified] notes.

    Proto on P3 (c_ = 256 x width 0.25 = 64, 32 prototypes at stride 4); cv4 per level Conv3x3 c4 -> Conv3x3 c4 ->
    1x1 32 coefficients, c4 = max(P3 channels / 4, 32); mask loss gain = box gain 7.5.
"""

import torch
import torch.nn as nn

from detectors.anchor_free.common import to_raw
from detectors.yolo11.modules import Conv
from detectors.yolo11_seg.build import NM, NPR, Proto, YOLO11Seg, _RecordingMatcher, split_seg_targets
from detectors.yolo11.build import make_divisible
from detectors.yolov8.build import VARIANT, YOLOv8, yolov8_cfg


class YOLOv8Seg(YOLOv8):
    mask_loss = YOLO11Seg.mask_loss
    decode_masks = YOLO11Seg.decode_masks

    def __init__(self, num_classes, variant=VARIANT):
        super().__init__(num_classes, variant)
        cfg = yolov8_cfg[variant]
        chs = list(self.body.out_dim)
        npr = make_divisible(min(NPR, cfg.get("max_channels", 1024)) * cfg["width"], 8)
        c4 = max(chs[0] // 4, NM)
        self.proto = Proto(chs[0], npr, NM)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, NM, 1)) for x in chs)
        self.criterion.matcher = _RecordingMatcher(self.criterion.matcher)
        self.mask_gain = self.criterion.loss_box_weight

    def forward(self, x, targets=None):
        feats = self.body(x)
        cls_feats, reg_feats = self.det_heads(feats)
        out = self.pred_layers(cls_feats, reg_feats)
        for k in ("anchors", "stride_tensor"):
            out[k] = [a.to(x.device) for a in out[k]]
        B = x.shape[0]
        coef = torch.cat([h(f).permute(0, 2, 3, 1).reshape(B, -1, NM) for h, f in zip(self.cv4, feats)], 1)
        proto = self.proto(feats[0])
        cls = torch.cat(out["pred_cls"], 1)
        raw = to_raw(torch.cat(out["pred_box"], 1), cls.sigmoid())
        if targets is None:
            return raw if self.training else (raw, {"coef": coef, "proto": proto})
        S = x.shape[2]
        tgt = split_seg_targets(targets, B, S)
        self.criterion.matcher.calls = []
        loss_dict, fg_mask, fg_labels = self.criterion(out, tgt)
        loss_mask = self.mask_loss(coef, proto, tgt, fg_mask.view(B, -1), S)
        loss_dict = {**loss_dict, "loss_mask": loss_mask}
        loss_dict["losses"] = loss_dict["losses"] + self.mask_gain * loss_mask
        flat = cls.reshape(-1, self.num_classes)
        self.stats.update(loss_dict, flat.sigmoid().max(-1).values, flat, fg_mask, fg_labels)
        return loss_dict["losses"], raw


def build_model(task, num_classes):
    return YOLOv8Seg(num_classes)
