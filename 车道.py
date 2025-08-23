import cv2
import numpy as np

def detect_lanes(image_path, output_path="lanes_detected.jpg"):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"未找到图片: {image_path}")
    original = img.copy()

    # 2. 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 高斯滤波去噪
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Canny边缘检测
    edges = cv2.Canny(blur, 50, 150)

    # 5. 限制区域（多边形掩膜），聚焦路面下方区域
    mask = np.zeros_like(edges)
    height, width = edges.shape
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height*0.6)),
        (0, int(height*0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 6. Hough变换检测直线
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=50)

    lane_count = 0
    if lines is not None:
        # 遍历每条检测到的线
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(original, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 绿色画线
        lane_count = len(lines) - 1 if len(lines) > 1 else 1  # 车道数量=线条数-1

    # 7. 显示并保存结果
    cv2.putText(original, f"Detected lanes: {lane_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(output_path, original)
    print(f"处理完成! 车道数量: {lane_count}, 结果已保存到 {output_path}")

# 示例调用
detect_lanes("road.jpg", "road_lanes.jpg")
