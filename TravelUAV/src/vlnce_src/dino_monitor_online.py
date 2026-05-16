import copy
import numpy as np
import math
import torch
from PIL import Image
import json
from src.common.param import model_args, args

# RGB_FOLDER = ['frontcamerarecord', 'downcamerarecord']


def _select_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _to_numpy_array(data):
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    if isinstance(data, Image.Image):
        return np.array(data)
    return np.asarray(data)

class DinoMonitor:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = DinoMonitor()
            return cls._instance
        return cls._instance
        
    def __init__(self, device=None):
        self.dino_model = None
        self.init_dino_model(device)
        self.object_desc_dict = dict()
        self.init_object_dict()
        
    def init_object_dict(self):
        with open(args.object_name_json_path, 'r') as f:
            file = json.load(f)
            for item in file:
                self.object_desc_dict[item['object_name']] = item['object_desc']
    
    def init_dino_model(self, device):
        import src.model_wrapper.utils.GroundingDINO as GroundingDINO
        import sys
        from functools import partial
        sys.path.append(GroundingDINO.__path__[0])
        from src.model_wrapper.utils.GroundingDINO.groundingdino.util.inference import load_model, predict
        device = _select_device(device)
        model = load_model(model_args.groundingdino_config, model_args.groundingdino_model_path)
        model.to(device=device)
        self.dino_model = partial(predict, model=model)
    
    def get_dino_results(self, episode, obj_info):
        images = episode[-1]['rgb_record']
        depths = episode[-1]['depth_record']
        done = False
        
        for i in range(len(images)):
            img = _to_numpy_array(images[i])
            depth = _to_numpy_array(depths[i])
            target_detections = []
            boxes, logits = self.detect(img, obj_info)

            if len(boxes) > 0:
                rgb_h, rgb_w = img.shape[:2]
                depth_h, depth_w = depth.shape[:2]
                scale_x = depth_w / float(rgb_w)
                scale_y = depth_h / float(rgb_h)

                for i, point in enumerate(boxes):
                    point = list(map(int, point))
                    center_x_rgb = int((point[0] + point[2]) / 2)
                    center_y_rgb = int((point[1] + point[3]) / 2)

                    center_x_depth = int(center_x_rgb * scale_x)
                    center_y_depth = int(center_y_rgb * scale_y)

                    center_x_depth = int(np.clip(center_x_depth, 0, depth_w - 1))
                    center_y_depth = int(np.clip(center_y_depth, 0, depth_h - 1))

                    depth_data = int(depth[center_y_depth, center_x_depth] / 2.55)
                    if depth_data < 18:
                        target_detections.append((float(logits[i]), depth_data))

            if len(target_detections) > 0:
                done = True
                break

        return done
    
    def detect(self, img, prompt):
        import groundingdino.datasets.transforms as T
        from groundingdino.util import box_ops
        
        img_src = _to_numpy_array(copy.deepcopy(img))
        if img_src.ndim != 3:
            raise ValueError(f'Expected HWC image, got shape={img_src.shape}')
        if img_src.dtype != np.uint8:
            img_src = np.clip(img_src, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_src)
        transform = T.Compose(
        [   T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_transformed, _ = transform(img_pil, None)
        boxes, logits, phrases = self.dino_model(
            image=image_transformed,
            caption=prompt,
            box_threshold=0.6,
            text_threshold=0.40
        )
        logits_np = _to_numpy_array(logits)
        H, W, _ = img_src.shape
        boxes_xyxy = (box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H], dtype=boxes.dtype, device=boxes.device)).cpu().numpy()

        keep_indices = []
        filtered_boxes = []
        for idx, box in enumerate(boxes_xyxy):
            if (box[2] - box[0]) / W > 0.6 or (box[3] - box[1]) / H > 0.5:
                continue
            keep_indices.append(idx)
            filtered_boxes.append(box)

        filtered_logits = logits_np[keep_indices] if len(keep_indices) > 0 else logits_np[:0]
        return filtered_boxes, filtered_logits
    
dino_monitor = DinoMonitor.get_instance()