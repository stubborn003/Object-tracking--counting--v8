import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# 加载预训练的 YOLO 模型
model = YOLO('yolov8n.pt')
# 输入与输出视频文件路径
VIDEO_PATH = "person_test3.mp4"
RESULT_PATH = "result1.mp4"

# 初始化视频写入器和跟踪历史记录
videowriter = None
track_history = defaultdict(lambda: [])  # 使用 defaultdict 来保存每个对象的跟踪历史
# 定义要跟踪的对象类别（根据YOLO类的索引）
OBJ_LIST = [0, 1, 2, 3, 5, 6, 7]

if __name__ == '__main__':
    # 打开输入视频文件
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():  # 检查视频文件是否成功打开
        print("Error opening video file.")
        exit()

        # 获取视频的帧率、宽和高
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 开始循环读取视频帧
    while True:
        success, frame = capture.read()  # 读取视频的一帧
        if not success:  # 检查是否成功读取
            print("读取帧失败")  # 如果读取失败，打印错误信息
            break

            # 使用 YOLO 模型对当前帧进行目标跟踪
        results = model.track(frame, persist=True, classes=OBJ_LIST)

        # 检查结果是否包含有效的检测框
        if results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            a_frame = results[0].plot(line_width=2)  # 绘制检测到的结果

            # 获取检测框的位置和跟踪 IDs
            boxes = results[0].boxes.xywh.cpu()  # 目标框的 (x, y, w, h)
            track_ids = results[0].boxes.id.int().cpu().tolist()  # 跟踪 ID 列表

            # 对每个检测到的目标进行处理
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box  # 解包检测框的坐标和宽高
                track = track_history[track_id]  # 获取当前跟踪 ID 的历史位置记录
                track.append((float(x), float(y)))  # 将当前位置添加到历史记录中
                if len(track) > 60:  # 限制历史记录的长度
                    track.pop(0)  # 移除最早的位置

                points = np.hstack(track).astype(np.int32).reshape(-1, 1, 2)  # 重塑为适合 OpenCV 的格式
                # 在当前帧上绘制对象的移动轨迹
                cv2.polylines(a_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)

        else:
            print("未检测到对象")  # 打印没有检测到对象的提示
            a_frame = frame  # 在没有检测到目标的情况下，显示原始帧

        # 初始化视频写入器（如果尚未创建）
        if videowriter is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 定义视频编码格式
            # 创建 VideoWriter 对象，用于将处理后的视频帧写入文件
            videowriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (frame_width, frame_height))

        videowriter.write(a_frame)  # 将当前帧写入输出视频文件
        cv2.imshow('yolov8 tracking', a_frame)  # 显示当前帧的跟踪结果

        # 如果按下 'q' 键，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            # 释放资源
    capture.release()  # 释放视频捕获对象
    videowriter.release()  # 释放视频写入器
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口

# names={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
# 1,数据来源source:图像路径、视频、目录、URL、设备ID
# 2.c0nf最小置信度：0.25
# 3.i0u交并比：0.7
# 4.max_det 允许的最大检测数 （默认：300）
# 5,classes:[2,3,7]只返回指定类别的检测结果
# 6,sh0W:窗口显示结果
# 7.save保存结果文件
# 8,save frames:逐帧分析视频，但是系统会报warning
# 9.saVe_txt:将检测结果保存成文本文件
#10.save_conf:可以在文本文件中保存置信度
#I1,shoW_labels:shoW为True的时候，是否显示标签
#12,sh0w_c0nf:是否显示置信度
#13.show_b0Xes:是否显示框
#14,Line_width:设置框框的宽度
