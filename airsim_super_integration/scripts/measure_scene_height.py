#!/usr/bin/env python3
import argparse
import os
import re
import time
from typing import Dict, Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def _parse_vector(text: str) -> Optional[List[float]]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 0:
        return None
    values: List[float] = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            return None
    return values


def parse_height_config(path: str) -> Dict[str, object]:
    """
    Lightweight config parser (regex-based) to quickly print height-related params.
    Not a full YAML parser; it only extracts a few keys if they appear in simple 'key: value' lines.
    """
    if not path or not os.path.exists(path):
        return {}
    data: Dict[str, object] = {}
    patterns = {
        "map_size": re.compile(r"^\s*map_size:\s*\[(.*)\]\s*$"),
        "virtual_ceil_height": re.compile(r"^\s*virtual_ceil_height:\s*([-\d\.]+)\s*$"),
        "virtual_ground_height": re.compile(r"^\s*virtual_ground_height:\s*([-\d\.]+)\s*$"),
        "resolution": re.compile(r"^\s*resolution:\s*([-\d\.]+)\s*$"),
        "map_voxel_num": re.compile(r"^\s*map_voxel_num:\s*\[(.*)\]\s*$"),
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for key, pat in patterns.items():
                m = pat.match(line)
                if not m:
                    continue
                if key in ("map_size", "map_voxel_num"):
                    vec = _parse_vector(m.group(1))
                    if vec is not None:
                        data[key] = vec
                else:
                    try:
                        data[key] = float(m.group(1))
                    except ValueError:
                        pass
    return data


class HeightProbe(Node):
    def __init__(self, topic: str, max_frames: int, duration: float, qos_reliability: str):
        super().__init__("scene_height_probe")
        self.min_z: Optional[float] = None
        self.max_z: Optional[float] = None
        self.frame_count = 0
        self.point_count = 0

        self.max_frames = max_frames
        self.duration = duration
        self.start_time = time.time()

        qos = QoSProfile(depth=10)
        qos.durability = QoSDurabilityPolicy.VOLATILE

        # Fix for your case: /cloud_registered publisher uses BEST_EFFORT.
        # Allow choosing via CLI too.
        if qos_reliability.lower() in ("best_effort", "besteffort", "be"):
            qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        elif qos_reliability.lower() in ("reliable", "rel"):
            qos.reliability = QoSReliabilityPolicy.RELIABLE
        else:
            self.get_logger().warn(
                f"Unknown --qos-reliability '{qos_reliability}', fallback to BEST_EFFORT"
            )
            qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.sub = self.create_subscription(PointCloud2, topic, self._callback, qos)

    def _callback(self, msg: PointCloud2):
        if self.frame_count >= self.max_frames:
            return

        # Collect z values
        zs: List[float] = []
        for p in point_cloud2.read_points(msg, field_names=("z",), skip_nans=True):
            zs.append(float(p[0]))

        self.frame_count += 1
        if not zs:
            return

        z_arr = np.asarray(zs, dtype=np.float32)
        z_min = float(np.min(z_arr))
        z_max = float(np.max(z_arr))

        if self.min_z is None or z_min < self.min_z:
            self.min_z = z_min
        if self.max_z is None or z_max > self.max_z:
            self.max_z = z_max

        self.point_count += len(zs)

    def done(self) -> bool:
        if self.frame_count >= self.max_frames:
            return True
        return (time.time() - self.start_time) >= self.duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cloud_registered")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--config",
        default="/mnt/data/super_ws/src/SUPER/super_planner/config/click_smooth_ros2.yaml",
    )
    parser.add_argument(
        "--qos-reliability",
        default="best_effort",
        choices=["best_effort", "reliable"],
        help="QoS reliability for subscription. Use best_effort for most PointCloud2 publishers.",
    )
    args = parser.parse_args()

    cfg = parse_height_config(args.config)
    if cfg:
        print(f"config_path: {args.config}")
        if "map_size" in cfg:
            print(f"map_size: {cfg['map_size']}")
        if "resolution" in cfg:
            print(f"resolution: {cfg['resolution']}")
        if "virtual_ground_height" in cfg and "virtual_ceil_height" in cfg:
            print(f"virtual_ground_height: {cfg['virtual_ground_height']}")
            print(f"virtual_ceil_height: {cfg['virtual_ceil_height']}")
    else:
        print(f"config_path_not_found: {args.config}")

    rclpy.init()
    node = HeightProbe(args.topic, args.frames, args.duration, args.qos_reliability)
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if node.min_z is None or node.max_z is None:
        print("no_points_received")
        print("hint: check publisher QoS via: ros2 topic info -v /cloud_registered")
        print("hint: try: --qos-reliability best_effort")
        return

    height_span = node.max_z - node.min_z
    print(f"cloud_min_z_enu: {node.min_z:.3f}")
    print(f"cloud_max_z_enu: {node.max_z:.3f}")
    print(f"cloud_height_span_enu: {height_span:.3f}")

    # If your world is actually NED, a common conversion is z_ned = -z_enu
    print(f"cloud_min_z_ned: {-node.max_z:.3f}")
    print(f"cloud_max_z_ned: {-node.min_z:.3f}")
    print(f"cloud_height_span_ned: {height_span:.3f}")
    print(f"frames: {node.frame_count}, points: {node.point_count}")


if __name__ == "__main__":
    main()