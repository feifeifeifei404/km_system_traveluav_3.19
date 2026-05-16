#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单独测试 AirSim record camera 图像 RPC 耗时。

测试内容：
1. FrontCameraRecord RGB
2. FrontCameraRecord DepthPerspective
3. DownCameraRecord RGB
4. DownCameraRecord DepthPerspective

用途：
判断 getImageResponsesForRecord() 里 4 合 1 请求到底是哪一个 camera / image type 慢。
"""

import argparse
import time
import sys

import airsim


def image_type_name(image_type):
    if image_type == airsim.ImageType.Scene:
        return "Scene/RGB"
    if image_type == airsim.ImageType.DepthPerspective:
        return "DepthPerspective"
    return str(image_type)


def print_response_diag(tag, response, cost):
    if response is None:
        print(f"[Record Split] DONE {tag} cost={cost:.3f}s response=None")
        return

    print(
        f"[Record Split] DONE {tag} cost={cost:.3f}s "
        f"width={response.width} height={response.height} "
        f"uint8_len={len(response.image_data_uint8)} "
        f"float_len={len(response.image_data_float)} "
        f"pixels_as_float={response.pixels_as_float} "
        f"compress={response.compress}"
    )


def test_one_request(client, vehicle_name, camera_name, image_type, pixels_as_float, compress):
    tag = (
        f"camera={camera_name} "
        f"type={image_type_name(image_type)} "
        f"pixels_as_float={pixels_as_float} "
        f"compress={compress}"
    )

    print("=" * 100)
    print(f"[Record Split] START {tag}")

    req = airsim.ImageRequest(
        camera_name,
        image_type,
        pixels_as_float,
        compress,
    )

    start = time.perf_counter()
    responses = client.simGetImages(
        requests=[req],
        vehicle_name=vehicle_name,
    )
    cost = time.perf_counter() - start

    if responses is None:
        print(f"[Record Split] FAILED {tag} cost={cost:.3f}s responses=None")
        return None, cost

    if len(responses) == 0:
        print(f"[Record Split] FAILED {tag} cost={cost:.3f}s responses=[]")
        return None, cost

    response = responses[0]
    print_response_diag(tag, response, cost)
    return response, cost


def test_camera_info(client, vehicle_name, camera_names):
    print("=" * 100)
    print("[CameraInfo] Checking record cameras")

    for camera_name in camera_names:
        try:
            start = time.perf_counter()
            info = client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name)
            cost = time.perf_counter() - start
            print(
                f"[CameraInfo] OK camera={camera_name} cost={cost:.3f}s "
                f"fov={info.fov} pose={info.pose}"
            )
        except Exception as exc:
            print(f"[CameraInfo] FAILED camera={camera_name}: {repr(exc)}")


def test_batch_four(client, vehicle_name, compress):
    print("=" * 100)
    print("[Batch4] START original 4-in-1 request")

    requests = [
        airsim.ImageRequest(
            "FrontCameraRecord",
            airsim.ImageType.Scene,
            False,
            compress,
        ),
        airsim.ImageRequest(
            "FrontCameraRecord",
            airsim.ImageType.DepthPerspective,
            True,
            compress,
        ),
        airsim.ImageRequest(
            "DownCameraRecord",
            airsim.ImageType.Scene,
            False,
            compress,
        ),
        airsim.ImageRequest(
            "DownCameraRecord",
            airsim.ImageType.DepthPerspective,
            True,
            compress,
        ),
    ]

    start = time.perf_counter()
    responses = client.simGetImages(
        requests=requests,
        vehicle_name=vehicle_name,
    )
    cost = time.perf_counter() - start

    if responses is None:
        print(f"[Batch4] FAILED cost={cost:.3f}s responses=None")
        return

    print(f"[Batch4] DONE cost={cost:.3f}s response_count={len(responses)}")

    for idx, response in enumerate(responses):
        if response is None:
            print(f"[Batch4] response[{idx}]=None")
            continue

        print(
            f"[Batch4] response[{idx}] "
            f"width={response.width} height={response.height} "
            f"uint8_len={len(response.image_data_uint8)} "
            f"float_len={len(response.image_data_float)} "
            f"pixels_as_float={response.pixels_as_float} "
            f"compress={response.compress}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25001)
    parser.add_argument("--vehicle_name", default="Drone_1")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--pause",
        action="store_true",
        help="测试前先 simPause(True)。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="测试前先 simPause(False)。",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="RGB/depth 请求使用 compress=True。默认 False，与原 getImageResponsesForRecord 一致。",
    )
    parser.add_argument(
        "--test_batch",
        action="store_true",
        help="最后额外测试原始 4 合 1 batch 请求。",
    )
    args = parser.parse_args()

    print("=" * 100)
    print(
        f"[Connect] ip={args.ip} port={args.port} "
        f"vehicle_name={args.vehicle_name} timeout={args.timeout}"
    )

    client = airsim.MultirotorClient(
        ip=args.ip,
        port=args.port,
        timeout_value=args.timeout,
    )
    client.confirmConnection()

    if args.resume:
        print("[AirSim] simPause(False)")
        client.simPause(False)
        time.sleep(0.2)

    if args.pause:
        print("[AirSim] simPause(True)")
        client.simPause(True)
        time.sleep(0.2)

    camera_names = ["FrontCameraRecord", "DownCameraRecord"]
    test_camera_info(client, args.vehicle_name, camera_names)

    tests = [
        (
            "FrontCameraRecord",
            airsim.ImageType.Scene,
            False,
            args.compress,
            "FrontCameraRecord RGB",
        ),
        (
            "FrontCameraRecord",
            airsim.ImageType.DepthPerspective,
            True,
            args.compress,
            "FrontCameraRecord Depth",
        ),
        (
            "DownCameraRecord",
            airsim.ImageType.Scene,
            False,
            args.compress,
            "DownCameraRecord RGB",
        ),
        (
            "DownCameraRecord",
            airsim.ImageType.DepthPerspective,
            True,
            args.compress,
            "DownCameraRecord Depth",
        ),
    ]

    summary = []

    for camera_name, image_type, pixels_as_float, compress, name in tests:
        try:
            _, cost = test_one_request(
                client=client,
                vehicle_name=args.vehicle_name,
                camera_name=camera_name,
                image_type=image_type,
                pixels_as_float=pixels_as_float,
                compress=compress,
            )
            summary.append((name, cost, "OK"))
        except Exception as exc:
            print(f"[Record Split] EXCEPTION {name}: {repr(exc)}")
            summary.append((name, None, f"EXCEPTION: {repr(exc)}"))

    if args.test_batch:
        try:
            test_batch_four(
                client=client,
                vehicle_name=args.vehicle_name,
                compress=args.compress,
            )
        except Exception as exc:
            print(f"[Batch4] EXCEPTION: {repr(exc)}")

    print("=" * 100)
    print("[Summary]")
    for name, cost, status in summary:
        if cost is None:
            print(f"{name:32s} {status}")
        else:
            print(f"{name:32s} {cost:10.3f}s {status}")

    print("=" * 100)
    print("[Done]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(130)