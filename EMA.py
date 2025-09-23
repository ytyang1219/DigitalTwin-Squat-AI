import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# 影片讀取
cap = cv2.VideoCapture("C:\\Users\\ytyan\\Desktop\\squat video\\30correct2.MP4")

raw_hip_y = []
ema_hip_y = []

EMA_WINDOW = 5
alpha = 2 / (EMA_WINDOW + 1)
c_lmlist = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if not results.pose_landmarks:
        continue

    # 原始 hip_y (取左右 hip 的平均 y，並乘上 frame 高度)
    h, w, _ = frame.shape
    hip_y = ((results.pose_landmarks.landmark[23].y +
              results.pose_landmarks.landmark[24].y) / 2) * h
    raw_hip_y.append(hip_y)

    # 收集 EMA window
    cur_lmlist = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
    c_lmlist.append(cur_lmlist)

    dived = (1 - (1 - alpha) ** EMA_WINDOW) / alpha

    if len(c_lmlist) >= EMA_WINDOW:
        ema_list = []
        for i, q in zip(range(EMA_WINDOW), range(EMA_WINDOW - 1, -1, -1)):
            temp = [list(map(lambda x: x * (1 - alpha) ** q / dived, c_lmlist[i][j])) for j in range(33)]
            ema_list.append(temp)
        arr = np.array(ema_list)
        s_arr = np.sum(arr, axis=0).tolist()

        # 更新 pose_landmarks
        for i in range(33):
            results.pose_landmarks.landmark[i].y = s_arr[i][1]

        hip_y_ema = ((results.pose_landmarks.landmark[23].y +
                      results.pose_landmarks.landmark[24].y) / 2) * h
        ema_hip_y.append(hip_y_ema)

        del c_lmlist[0]

cap.release()

# 繪製結果比較（兩張圖分開）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10), sharex=True)

ax1.plot(raw_hip_y, color="blue")
ax1.set_title("Original Hip Y")

ax2.plot(range(EMA_WINDOW-1, EMA_WINDOW-1+len(ema_hip_y)), ema_hip_y, color="orange")
ax2.set_title("EMA Smooth Hip Y")

plt.xlabel("Frame")
plt.show()