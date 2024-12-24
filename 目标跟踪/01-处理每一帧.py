import cv2

import numpy as np

VIDEO_PATH ="car_test2.mp4"
RESULT_PATH ="result1.mp4"
videowriter=None
polygonPoints=np.array([[450,350],[650,350],[800,400],[600,400]],dtype=np.int32)
if __name__=='__main__':
    capture=cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()
    # frame_width=capture.get(3)#宽度
    # frame_height=capture.get(4)#高度
    # fps=capture.get(5)#帧
    fps = capture.get(cv2.CAP_PROP_FPS)  # 获取帧率
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获取宽度
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获取高度
    while True:
        success,frame=capture.read()
        if not success :
            print("读取帧失败")
            break

        cv2.polylines(frame,[polygonPoints],True,(255,0,0),2)#多边形
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask,[polygonPoints],(255,0,0))
        frame=cv2.addWeighted(frame,0.7,mask,0.3,0)

        cv2.line(frame,(0,int(frame_height/2)),(int(frame_width),int(frame_height/2)),(0,0,255),1)
        if videowriter is None:
            fourcc=cv2.VideoWriter_fourcc("m","p","4","v")
            videowriter =cv2.VideoWriter(RESULT_PATH,fourcc,fps
                                         ,(int(frame_width),int(frame_height)))
        videowriter.write(frame)
        # 缩放帧
        frame = cv2.resize(frame, (1080, int(1080 * frame_height / frame_width)))
        cv2.imshow("frame",frame)
        cv2.waitKey(1)

    capture.release()
    videowriter.release()
    cv2.destroyAllWindows()