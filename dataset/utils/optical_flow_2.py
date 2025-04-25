import cv2
import numpy as np

# 全局变量，用于保存鼠标点击的点
selected_points = []


# 鼠标回调函数，点击时获取坐标
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("First Frame", param)


def process_video(video_path):
    global selected_points
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = int(1000 / fps)  # 每帧显示的时间（毫秒）

    # 读取第一帧
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频文件。")
        return

    # 创建一个用于显示和选择特征点的副本
    first_frame_copy = first_frame.copy()

    # 显示第一帧，用户可以点击选择特征点
    cv2.imshow("First Frame", first_frame_copy)
    cv2.setMouseCallback("First Frame", select_point, first_frame_copy)

    print("请点击选择特征点，按 Enter 键确认")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter 键确认选择
            break

    cv2.destroyWindow("First Frame")

    # 将鼠标点击的点转换为np数组
    if len(selected_points) == 0:
        print("没有选择任何特征点。")
        return
    selected_points = np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)

    # 将第一帧转换为灰度图像
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # 设置用于绘制跟踪点的颜色
    color = np.random.randint(0, 255, (len(selected_points), 3))

    while True:
        # 读取当前帧
        ret, frame = cap.read()
        if not ret:
            break

        # 将当前帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用光流法计算特征点在当前帧中的位置
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, selected_points, None)

        # 选择成功跟踪的点
        good_new = next_pts[status == 1]
        good_old = selected_points[status == 1]

        # 在每一帧中绘制当前帧的跟踪点和轨迹
        frame_copy = frame.copy()
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # 画线连接旧点和新点
            frame_copy = cv2.line(frame_copy, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            # 在新位置画一个圆圈
            frame_copy = cv2.circle(frame_copy, (int(a), int(b)), 5, color[i].tolist(), -1)

        # 显示每一帧的光流结果
        cv2.imshow('Optical Flow - LK', frame_copy)

        # 按 'q' 键退出
        if cv2.waitKey(frame_time) & 0xFF == ord('q'):
            break

        # 更新前一帧和特征点
        prev_gray = gray.copy()
        selected_points = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_video('2024-04-27-16-33-04_wxy-f.mp4')
