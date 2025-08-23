# -*- coding: utf-8 -*-
"""
YOLO 实时检测 + 跟踪 + 运动判定（车辆 & 行人）
- 绿色框：静止/缓慢
- 红色框：在移动
依赖：pip install ultralytics opencv-python
运行示例：
  python detect_track_motion.py --source 0
  python detect_track_motion.py --source path/to/video.mp4 --model yolov8s.pt --move_thresh_pxps 60
"""

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# 关注的类别：行人 + 各类车辆
DEFAULT_TARGET_CLASS_NAMES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="视频源：'0' 为摄像头，或视频文件路径")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO 模型文件，如 yolov8n.pt / yolov8s.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IOU 阈值")
    ap.add_argument("--move_window", type=int, default=5, help="计算速度的滑动窗口帧数")
    ap.add_argument("--move_thresh_pxps", type=float, default=50.0,
                    help="移动判定阈值（像素/秒，px/s），建议随分辨率/场景调节")
    ap.add_argument("--show_fps", action="store_true", help="在画面左上角显示 FPS")
    args = ap.parse_args()

    # 摄像头编号
    if args.source.strip().isdigit():
        args.source = int(args.source)
    return args

def main():
    args = parse_args()

    model = YOLO(args.model)

    # 打开视频源，以便我们自己控制 FPS 估计与显示
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源：{args.source}")

    # 读取类别名
    names = model.model.names if hasattr(model, "model") else model.names
    target_class_ids = {i for i, n in names.items() if n in DEFAULT_TARGET_CLASS_NAMES}

    # 轨迹历史：每个 track_id 存最近若干 (timestamp, cx, cy)
    history = defaultdict(lambda: deque(maxlen=50))  # 适当留长，便于窗口取最近 K 帧
    last_time = time.time()
    fps_smooth = None

    # 用 YOLO 的流式 track（ByteTrack），我们自己喂帧，便于叠加绘制
    # persist=True 保持追踪 ID 连续；tracker 使用内置 bytetrack.yaml
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        # 推理 + 跟踪
        results = model.track(
            source=frame,
            stream=True,            # 返回生成器
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )

        # YOLO stream=True 时会返回一个迭代器，这里只取当前帧的第一份结果
        try:
            res = next(results)
        except StopIteration:
            res = None

        now = time.time()
        dt = now - last_time
        last_time = now

        # 估计 FPS（指数滑动平均）
        if dt > 0:
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)

        if res is not None and res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.array([None] * len(boxes))

            for i in range(len(boxes)):
                if cls[i] not in target_class_ids:
                    continue

                x1, y1, x2, y2 = xyxy[i].astype(int)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w / 2.0, y1 + h / 2.0

                track_id = ids[i] if i < len(ids) else None
                class_name = names.get(int(cls[i]), str(cls[i]))
                label = f"{class_name} {conf[i]:.2f}"
                speed_pxps = 0.0
                is_moving = False

                if track_id is not None:
                    # 更新历史
                    history[track_id].append((now, cx, cy))

                    # 取最近 move_window 帧估计速度（像素/秒）
                    hist = history[track_id]
                    if len(hist) >= 2:
                        # 只取最近 K 个点
                        k = min(args.move_window, len(hist) - 1)
                        t_old, x_old, y_old = hist[-1 - k]
                        t_new, x_new, y_new = hist[-1]
                        dt_hist = max(t_new - t_old, 1e-6)
                        dist = ((x_new - x_old) ** 2 + (y_new - y_old) ** 2) ** 0.5
                        speed_pxps = dist / dt_hist
                        is_moving = speed_pxps >= args.move_thresh_pxps
                        label = f"ID {track_id} {class_name} {speed_pxps:.1f}px/s"

                # 颜色：移动=红，不动=绿（BGR）
                color = (0, 0, 255) if is_moving else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 画质心
                cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)

                # 放置标签
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if args.show_fps and fps_smooth is not None:
            fps_text = f"FPS: {fps_smooth:.1f}"
            cv2.putText(frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("YOLO Track & Motion (q 退出)", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
