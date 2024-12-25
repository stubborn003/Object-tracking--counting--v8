import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

model = YOLO('yolov8n.pt')  # 加载 YOLO 模型
VIDEO_PATH = "person_test2.mp4"  # 输入视频路径
RESULT_PATH = "result1.mp4"  # 输出视频路径

videowriter = None  # 初始化视频写入器
track_history = defaultdict(lambda: [])  # 创建一个默认字典用于存储每个物体的轨迹
OBJ_LIST = [0, 1, 2, 3, 5, 6, 7]  # 要跟踪的物体类别
count_passed = 0  # 进入区域的物体计数
count_exited = 0  # 离开区域的物体计数
entered_ids = set()  # 存储已进入区域的物体 ID
exited_ids = set()  # 存储已离开区域的物体 ID

# 定义多边形区域的点
polygonPoints = np.array([[100, 500], [900, 500], [900, 800], [100, 800]], np.int32)

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

        # 绘制多边形区域掩膜
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [polygonPoints], (255, 0, 0))  # 用红色填充多边形

        # 将掩膜应用到当前帧上
        a_frame = cv2.addWeighted(a_frame, 1, mask, 0.5, 0)  # 将掩膜叠加到当前帧
        # 检查是否检测到物体
        if results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()  # 获取检测框的坐标 (x, y, w, h)
            track_ids = results[0].boxes.id.int().cpu().tolist()  # 获取每个检测框的跟踪 ID

            for box, track_id in zip(boxes, track_ids):  # 遍历检测框和对应的跟踪 ID
                x, y, w, h = box  # 解包检测框的坐标和宽高
                center = np.array([x + w / 2, y + h / 2], dtype=np.float32)  # 计算检测框中心的坐标并转换为 NumPy 数组

                track = track_history[track_id]  # 获取当前跟踪 ID 的轨迹
                track.append((float(x), float(y)))  # 将当前框的坐标添加到轨迹中
                if len(track) > 60:  # 如果轨迹超过 60 个点，移除最早的一个
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape(-1, 1, 2)  # 将轨迹转换为 NumPy 数组
                cv2.polylines(a_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)  # 绘制轨迹

                # 检查物体是否进入或离开区域
                if track_id not in entered_ids and cv2.pointPolygonTest(polygonPoints, center, False) >= 0:
                    count_passed += 1  # 进入区域计数
                    entered_ids.add(track_id)  # 标记为已进入
                elif track_id in entered_ids and cv2.pointPolygonTest(polygonPoints, center, True) < 0:
                    count_exited += 1  # 离开区域计数
                    entered_ids.remove(track_id)  # 从已进入集合中移除
                    exited_ids.add(track_id)  # 标记为已离开

        # 显示计数结果
        cv2.putText(a_frame, f'Count Entered: {count_passed}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(a_frame, f'Count Exited: {count_exited}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if videowriter is None:  # 如果视频写入器尚未初始化
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # 定义视频编码格式
            videowriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps,
                                          (int(frame_width), int(frame_height)))  # 初始化视频写入器

        videowriter.write(a_frame)  # 写入当前帧到输出视频
        cv2.imshow('yolov8 tracking', a_frame)  # 显示当前帧的跟踪结果
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 检查是否按下 'q' 键
            break  # 跳出循环

    capture.release()  # 释放视频捕获对象
    videowriter.release()  # 释放视频写入器
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口
