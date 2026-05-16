#!/usr/bin/env python3
import threading
import time
from pathlib import Path

import airsim
import cv2
import numpy as np


IP = "127.0.0.1"
PORTS = [25001]
VEHICLE = "Drone_1"

CAMERAS = [
    "FrontCameraRecord",
    "DownCameraRecord",
]

DEPTH_CASES = [
    ("DepthPerspective", airsim.ImageType.DepthPerspective, True, False),
    ("DepthPlanar", airsim.ImageType.DepthPlanar, True, False),
    ("DepthPerspective_uint8", airsim.ImageType.DepthPerspective, False, False),
    ("DepthPlanar_uint8", airsim.ImageType.DepthPlanar, False, False),
]

OUT_DIR = Path("./record_depth_compare_outputs")
OUT_DIR.mkdir(exist_ok=True)


class Capture:
    def __init__(self, camera, name, image_type, pixels_as_float, compress, response, elapsed):
        self.camera = camera
        self.name = name
        self.image_type = image_type
        self.pixels_as_float = pixels_as_float
        self.compress = compress
        self.response = response
        self.elapsed = elapsed
        self.width = int(response.width)
        self.height = int(response.height)

    @property
    def resolution(self):
        return f"{self.width}x{self.height}"


def connect_airsim():
    last_err = None
    for port in PORTS:
        try:
            print(f"[CONNECT] Trying {IP}:{port}")
            client = airsim.MultirotorClient(ip=IP, port=port, timeout_value=300)
            if client.ping():
                print(f"[CONNECT] Connected on port {port}")
                client.confirmConnection()
                return client, port
        except Exception as e:
            print(f"[CONNECT] Failed on port {port}: {repr(e)}")
            last_err = e

    raise RuntimeError(f"Cannot connect to AirSim on {PORTS}. Last error: {last_err}")


def make_request(camera, image_type, pixels_as_float, compress=False):
    return airsim.ImageRequest(
        camera,
        image_type,
        pixels_as_float=pixels_as_float,
        compress=compress,
    )


def get_image(client, camera, name, image_type, pixels_as_float, compress=False):
    req = make_request(camera, image_type, pixels_as_float, compress)

    t0 = time.perf_counter()
    resp = client.simGetImages([req], vehicle_name=VEHICLE)
    dt = time.perf_counter() - t0

    if not resp:
        raise RuntimeError(
            f"Empty response: camera={camera}, type={name}, "
            f"float={pixels_as_float}, compress={compress}"
        )

    capture = Capture(camera, name, image_type, pixels_as_float, compress, resp[0], dt)
    print_capture(capture)
    return capture


def get_images_batch(client, requests, label):
    t0 = time.perf_counter()
    responses = client.simGetImages([item["request"] for item in requests], vehicle_name=VEHICLE)
    dt = time.perf_counter() - t0
    if not responses or len(responses) != len(requests):
        raise RuntimeError(f"Invalid batch response for {label}: got={0 if not responses else len(responses)}")

    captures = []
    for item, response in zip(requests, responses):
        captures.append(
            Capture(
                item["camera"],
                item["name"],
                item["image_type"],
                item["pixels_as_float"],
                item["compress"],
                response,
                dt,
            )
        )
    print(f"[BATCH RESULT] {label}: total_time={dt:.3f}s requests={len(requests)}")
    for capture in captures:
        print_capture(capture, prefix="  [BATCH ITEM]")
    return captures, dt


def get_images_parallel(client, request_items, label):
    results = [None] * len(request_items)
    errors = [None] * len(request_items)
    port = getattr(client, "_port", PORTS[0])

    def worker(index, item):
        worker_client = None
        try:
            worker_client = airsim.MultirotorClient(ip=IP, port=port, timeout_value=300)
            if not worker_client.ping():
                raise RuntimeError(f"worker client ping failed on port={port}")
            results[index] = get_image(
                worker_client,
                item["camera"],
                item["name"],
                item["image_type"],
                item["pixels_as_float"],
                item["compress"],
            )
        except Exception as e:
            errors[index] = e
        finally:
            if worker_client is not None:
                try:
                    worker_client.close()
                except Exception:
                    pass

    t0 = time.perf_counter()
    threads = []
    for idx, item in enumerate(request_items):
        thread = threading.Thread(target=worker, args=(idx, item), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    dt = time.perf_counter() - t0

    if any(errors):
        raise RuntimeError(f"Parallel errors for {label}: {errors}")

    print(f"[PARALLEL RESULT] {label}: wall_time={dt:.3f}s requests={len(request_items)}")
    return results, dt


def print_capture(capture, prefix="[RESULT]"):
    r = capture.response
    print(
        f"{prefix} camera={capture.camera:20s} "
        f"case={capture.name:24s} "
        f"type={int(capture.image_type)} "
        f"float={str(capture.pixels_as_float):5s} "
        f"compress={str(capture.compress):5s} "
        f"time={capture.elapsed:8.3f}s "
        f"w={r.width:4d} h={r.height:4d} "
        f"uint8_len={len(r.image_data_uint8):9d} "
        f"float_len={len(r.image_data_float):9d}"
    )


def save_scene_uint8(capture):
    resp = capture.response
    data = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    img = data.reshape(resp.height, resp.width, 3)
    path = OUT_DIR / f"{capture.camera}_Scene_{capture.resolution}.png"
    cv2.imwrite(str(path), img)
    print(f"[SAVE] {path}")


def save_uint8_image(capture):
    resp = capture.response
    data = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)

    expected_3ch = resp.height * resp.width * 3
    expected_1ch = resp.height * resp.width

    if len(data) == expected_3ch:
        img = data.reshape(resp.height, resp.width, 3)
    elif len(data) == expected_1ch:
        img = data.reshape(resp.height, resp.width)
    else:
        raise RuntimeError(
            f"Unexpected uint8 length={len(data)}, "
            f"expected {expected_1ch} or {expected_3ch}"
        )

    path = OUT_DIR / f"{capture.camera}_{capture.name}_{capture.resolution}.png"
    cv2.imwrite(str(path), img)

    print(
        f"[SAVE] {path} "
        f"shape={img.shape} min={int(np.min(img))} "
        f"max={int(np.max(img))} mean={float(np.mean(img)):.3f}"
    )


def depth_array(capture):
    resp = capture.response
    depth = np.asarray(resp.image_data_float, dtype=np.float32)
    return depth.reshape(resp.height, resp.width)


def save_float_depth(capture, clip_max=80.0):
    depth = depth_array(capture)
    prefix = OUT_DIR / f"{capture.camera}_{capture.name}_floatTrue_{capture.resolution}"

    npy_path = str(prefix) + ".npy"
    vis_path = str(prefix) + "_vis.png"
    np.save(npy_path, depth)

    valid = np.isfinite(depth)
    if np.any(valid):
        d_valid = depth[valid]
        print(
            f"[DEPTH STATS] {prefix.name}: "
            f"min={float(np.min(d_valid)):.4f}, "
            f"max={float(np.max(d_valid)):.4f}, "
            f"mean={float(np.mean(d_valid)):.4f}, "
            f"median={float(np.median(d_valid)):.4f}, "
            f"finite_ratio={np.count_nonzero(valid) / depth.size:.4f}"
        )
    else:
        print(f"[DEPTH STATS] {prefix.name}: no finite values")

    vis = np.clip(depth, 0, clip_max)
    vis = (vis / clip_max * 255.0).astype(np.uint8)
    cv2.imwrite(vis_path, vis)

    print(f"[SAVE] {npy_path}")
    print(f"[SAVE] {vis_path}")

    if capture.width != 1024 or capture.height != 1024:
        upscaled_path = str(prefix) + "_vis_UPSCALED_TO_1024.png"
        upscaled = cv2.resize(vis, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(upscaled_path, upscaled)
        print(f"[SAVE] {upscaled_path}")

    return depth


def robust_stats(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def compare_float_depths(camera, captures):
    float_caps = [c for c in captures if c.pixels_as_float and len(c.response.image_data_float) > 0]
    by_res = {(c.width, c.height): c for c in float_caps}

    if (512, 512) not in by_res or (1024, 1024) not in by_res:
        print(
            f"[COMPARE][{camera}] Need both 512x512 and 1024x1024 float depth. "
            f"Available: {sorted([c.resolution for c in float_caps])}"
        )
        return

    cap512 = by_res[(512, 512)]
    cap1024 = by_res[(1024, 1024)]

    d512 = depth_array(cap512)
    d1024 = depth_array(cap1024)
    d512_up = cv2.resize(d512, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    raw_diff = np.abs(d512_up - d1024)
    valid = np.isfinite(raw_diff)
    finite_diff = raw_diff[valid]

    if finite_diff.size:
        print(f"[COMPARE][{camera}] all finite diff stats: {robust_stats(finite_diff)}")

    bounded = valid & np.isfinite(d512_up) & np.isfinite(d1024) & (d512_up < 100.0) & (d1024 < 100.0)
    bounded_diff = raw_diff[bounded]
    if bounded_diff.size:
        print(
            f"[COMPARE][{camera}] bounded depth<100m diff stats: "
            f"{robust_stats(bounded_diff)} valid_ratio={bounded_diff.size / raw_diff.size:.4f}"
        )

    near = bounded & (np.minimum(d512_up, d1024) < 20.0)
    near_diff = raw_diff[near]
    if near_diff.size:
        print(
            f"[COMPARE][{camera}] near depth<20m diff stats: "
            f"{robust_stats(near_diff)} valid_ratio={near_diff.size / raw_diff.size:.4f}"
        )

    diff_vis = np.clip(raw_diff, 0, 10) / 10 * 255
    diff_path = OUT_DIR / f"{camera}_DIFF_512up_vs_1024_abs_clip10m.png"
    cv2.imwrite(str(diff_path), diff_vis.astype(np.uint8))
    print(f"[SAVE] {diff_path}")


def build_record_depth_512_requests():
    return [
        {
            "camera": camera,
            "name": "DepthPerspective_512_float",
            "image_type": airsim.ImageType.DepthPerspective,
            "pixels_as_float": True,
            "compress": False,
            "request": make_request(camera, airsim.ImageType.DepthPerspective, True, False),
        }
        for camera in CAMERAS
    ]


def benchmark_parallel_vs_sequential(client):
    print("\n" + "=" * 100)
    print("Record depth 512 sequential vs parallel benchmark")
    print("=" * 100)

    requests = build_record_depth_512_requests()

    seq_start = time.perf_counter()
    seq_caps = []
    for item in requests:
        seq_caps.append(
            get_image(
                client,
                item["camera"],
                item["name"],
                item["image_type"],
                item["pixels_as_float"],
                item["compress"],
            )
        )
    seq_time = time.perf_counter() - seq_start
    print(f"[SEQUENTIAL RESULT] total_time={seq_time:.3f}s")

    par_caps, par_time = get_images_parallel(client, requests, "record depth 512 two cameras")

    batch_caps, batch_time = get_images_batch(client, requests, "record depth 512 two cameras")

    print("\n[SUMMARY] record depth 512")
    print(f"  sequential_total={seq_time:.3f}s")
    print(f"  parallel_wall   ={par_time:.3f}s speedup={seq_time / max(par_time, 1e-9):.2f}x")
    print(f"  batch_total     ={batch_time:.3f}s speedup={seq_time / max(batch_time, 1e-9):.2f}x")

    return seq_caps + par_caps + batch_caps


def main():
    client, port = connect_airsim()
    print(f"[INFO] using AirSim port={port}")

    client.simPause(True)
    print("[INFO] simPause=True")

    benchmark_parallel_vs_sequential(client)

    print("\n" + "=" * 100)
    print("Record camera 512 vs 1024 depth comparison")
    print("=" * 100)

    for camera in CAMERAS:
        print("\n" + "#" * 100)
        print(f"[CAMERA] {camera}")
        print("#" * 100)

        captures = []

        scene_cap = get_image(
            client,
            camera=camera,
            name="Scene",
            image_type=airsim.ImageType.Scene,
            pixels_as_float=False,
            compress=False,
        )
        save_scene_uint8(scene_cap)

        for name, image_type, pixels_as_float, compress in DEPTH_CASES:
            cap = get_image(
                client,
                camera=camera,
                name=name,
                image_type=image_type,
                pixels_as_float=pixels_as_float,
                compress=compress,
            )
            captures.append(cap)

            if pixels_as_float:
                save_float_depth(cap)
            else:
                save_uint8_image(cap)

        compare_float_depths(camera, captures)

    client.simPause(True)
    print("\n[DONE] outputs saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
