import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

model = YOLO('yolov8n.pt')  # 加载 YOLO 模型
VIDEO_PATH = "car_test1.mp4"  # 输入视频路径
RESULT_PATH = "result1.mp4"  # 输出视频路径

videowriter = None  # 初始化视频写入器
track_history = defaultdict(lambda: [])  # 创建一个默认字典用于存储每个物体的轨迹
OBJ_LIST = [0, 1, 2, 3, 5, 6, 7]  # 要跟踪的物体类别
count = 0  # 经过线的物体计数
passed_ids = set()  # 存储已经经过线的物体 ID
line_y = 700  # 线的 Y 坐标

if __name__ == '__main__':
    capture = cv2.VideoCapture(VIDEO_PATH)  # 打开视频文件
    if not capture.isOpened():  # 检查视频文件是否成功打开
        print("Error opening video file.")
        exit()  # 退出程序
    fps = capture.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获取视频帧宽度
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获取视频帧高度

    while True:  # 循环读取视频帧
        success, frame = capture.read()  # 读取一帧视频
        if not success:  # 检查是否成功读取帧
            print("读取帧完成")
            break  # 跳出循环

        results = model.track(frame, persist=True, classes=OBJ_LIST)  # 使用 YOLO 模型进行跟踪
        a_frame = results[0].plot(line_width=2) if results[0] is not None else frame  # 绘制检测结果

        # 绘制水平线
        cv2.line(a_frame, (0, line_y), (int(frame_width), line_y), (0, 255, 0), 2)  # 绿色线

        # 检查是否检测到物体
        if results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()  # 获取检测框的坐标 (x, y, w, h)
            track_ids = results[0].boxes.id.int().cpu().tolist()  # 获取每个检测框的跟踪 ID

            for box, track_id in zip(boxes, track_ids):  # 遍历检测框和对应的跟踪 ID
                x, y, w, h = box  # 解包检测框的坐标和宽高
                center_y = y + h / 2  # 计算检测框中心的 Y 坐标

                track = track_history[track_id]  # 获取当前跟踪 ID 的轨迹
                track.append((float(x), float(y)))  # 将当前框的坐标添加到轨迹中
                if len(track) > 60:  # 如果轨迹超过 60 个点，移除最早的一个
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape(-1, 1, 2)  # 将轨迹转换为 NumPy 数组
                cv2.polylines(a_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)  # 绘制轨迹

                # 检查物体是否经过线
                if track_id not in passed_ids:  # 如果该 ID 还没有通过线
                    if center_y > line_y:  # 如果当前 Y 坐标在线的下方
                        if len(track) >= 2 and track[-2][1] <= line_y:  # 确保有足够的历史点
                            count += 1  # 增加经过线的计数
                            passed_ids.add(track_id)  # 标记为已通过

        # 显示计数结果
        cv2.putText(a_frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if videowriter is None:  # 如果视频写入器尚未初始化
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # 定义视频编码格式
            videowriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps,
                                          (int(frame_width), int(frame_height)))  # 初始化视频写入器

        videowriter.write(frame)  # 写入当前帧到输出视频
        cv2.imshow('yolov8 tracking', a_frame)  # 显示当前帧的跟踪结果
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 检查是否按下 'q' 键
            break  # 跳出循环

    capture.release()  # 释放视频捕获对象
    videowriter.release()  # 释放视频写入器
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口
