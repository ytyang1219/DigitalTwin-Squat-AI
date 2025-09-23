import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

def test_squat_lowest_to_excel(video_path, output_excel="test.xlsx"):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("無法開啟影片")
        return

    FRAME_SMOOTH = 5
    MIN_SQUAT_DURATION = 10
    HIP_THRESHOLD_RATIO = 0.05
    DEPTH_THRESHOLD_RATIO = 0.25
    EMA_WINDOW = 5
    alpha = 2 / (EMA_WINDOW + 1)
    dived = 1 * (1 - (1 - alpha) ** EMA_WINDOW) / (1 - (1 - alpha))  # 等比數列和

    state = "idle"
    squat_id = 0
    c_lmlist = []
    history = []
    segment = []
    lowest_points_data = []
    frame_idx = 0
    frame_skip = 2
    fixed_leg_length = None
    HIP_Y_THRESHOLD = None
    MIN_SQUAT_DEPTH = None
    min_hip_y = None
    max_hip_y = None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 只處理每 frame_skip 幀
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if not (results.pose_landmarks and results.pose_world_landmarks):
            continue

        cur_lmlist = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        c_lmlist.append(cur_lmlist)

        if len(c_lmlist) >= EMA_WINDOW:
            ema_list = []
            for i, q in zip(range(EMA_WINDOW), range(EMA_WINDOW - 1, -1, -1)):
                temp = []
                for j in range(33):
                    ema = list(map(lambda x: x * (1 - alpha) ** q / dived, c_lmlist[i][j]))
                    temp.append(ema)
                ema_list.append(temp)
                
            arr = np.array(ema_list)
            s_arr = np.sum(arr, axis=0).tolist()
            
            for i in range(33):
                results.pose_landmarks.landmark[i].x = s_arr[i][0]
                results.pose_landmarks.landmark[i].y = s_arr[i][1]
                results.pose_landmarks.landmark[i].z = s_arr[i][2]
                results.pose_landmarks.landmark[i].visibility = s_arr[i][3]
            del c_lmlist[0]

        hip_y = ((results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y) / 2) * frame_height
        history.append((frame.copy(), hip_y, results.pose_landmarks.landmark, results.pose_world_landmarks.landmark))

        # 臉部可見性檢查（每次都要做）
        face_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        face_visible = all(results.pose_landmarks.landmark[i].visibility > 0.98 for i in face_ids)

        # 若腿長尚未初始化，嘗試做腿部伸直與初始化
        def get_slope(p1, p2):
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            if abs(dx) < 1e-6:
                return float('inf')
            return dy / dx

        l_hip = results.pose_landmarks.landmark[23]
        l_knee = results.pose_landmarks.landmark[25]
        l_ankle = results.pose_landmarks.landmark[27]
        slope_thigh = get_slope(l_hip, l_knee)
        slope_shank = get_slope(l_knee, l_ankle)
        slope_close = abs(slope_thigh - slope_shank) < 3

        print(f"[DEBUG] slope_thigh = {slope_thigh:.2f}, slope_shank = {slope_shank:.2f}, 差值 = {abs(slope_thigh - slope_shank):.2f}")
        print(f"[DEBUG] 嘗試初始化腿長 → face_visible: {face_visible}, slope_close: {slope_close}")

        if fixed_leg_length is None:
            if not slope_close:
                print("[DEBUG] 腿未打直 → 初始化腿長失敗")
            elif face_visible:
                r_leg = np.linalg.norm([frame_width * (results.pose_landmarks.landmark[24].x - results.pose_landmarks.landmark[26].x),
                                        frame_height * (results.pose_landmarks.landmark[24].y - results.pose_landmarks.landmark[26].y)])
                l_leg = np.linalg.norm([frame_width * (results.pose_landmarks.landmark[25].x - results.pose_landmarks.landmark[23].x),
                                        frame_height * (results.pose_landmarks.landmark[25].y - results.pose_landmarks.landmark[23].y)])
                fixed_leg_length = (r_leg + l_leg) / 2
                HIP_Y_THRESHOLD = fixed_leg_length * HIP_THRESHOLD_RATIO
                MIN_SQUAT_DEPTH = fixed_leg_length * DEPTH_THRESHOLD_RATIO
                print(f"[DEBUG] ✅ 腿長初始化成功，固定腿長: {fixed_leg_length:.2f}")

        # 平滑 hip_y
        smoothed_hip_y = np.mean([h[1] for h in history[-FRAME_SMOOTH:]]) if len(history) >= FRAME_SMOOTH else hip_y

        # 判斷是否進入 squatting
        if state == "idle" and len(history) >= MIN_SQUAT_DURATION and HIP_Y_THRESHOLD:
            prev_hip_y = np.mean([h[1] for h in history[-MIN_SQUAT_DURATION:-MIN_SQUAT_DURATION//2]])
            delta_hip_y = smoothed_hip_y - prev_hip_y
            print(f"[DEBUG] 嘗試進入 squatting ➜ Δhip_y = {delta_hip_y:.2f}, 門檻 = {HIP_Y_THRESHOLD:.2f}")
            if face_visible and delta_hip_y > HIP_Y_THRESHOLD:
                print("[DEBUG] ✅ hip_y 落差足夠，進入 squatting")
                state = "squatting"
                segment = []
                min_hip_y = smoothed_hip_y
                max_hip_y = prev_hip_y
            else:
                print("[DEBUG] ❌ 未達入深蹲條件")
        elif state == "squatting":
            segment.append((frame.copy(), hip_y, results.pose_landmarks.landmark, results.pose_world_landmarks.landmark))
            min_hip_y = min(min_hip_y, smoothed_hip_y)
            max_hip_y = max(max_hip_y, smoothed_hip_y)

            if len(history) >= MIN_SQUAT_DURATION:
                prev_hip_y = np.mean([h[1] for h in history[-MIN_SQUAT_DURATION:-MIN_SQUAT_DURATION//2]])
                if prev_hip_y - smoothed_hip_y > HIP_Y_THRESHOLD and (max_hip_y - min_hip_y) > MIN_SQUAT_DEPTH:
                    lowest = max(segment, key=lambda x: x[1])
                    points = lowest[2]
                    key_indices = [23, 24, 25, 26, 29, 30, 31, 32]
                    for idx in key_indices:
                        lowest_points_data.append({
                            "squat_id": squat_id + 1,
                            "point_index": idx,
                            "x": points[idx].x,
                            "y": points[idx].y,
                            "z": points[idx].z
                        })
                    squat_id += 1
                    state = "idle"

    cap.release()

    if lowest_points_data:
        df = pd.DataFrame(lowest_points_data)
        df.to_excel(output_excel, index=False)
        print(f"✅ 測試完成，結果已輸出至 {output_excel}")
    else:
        print("⚠️ 未偵測到深蹲動作")

# 測試呼叫
if __name__ == "__main__":
    test_squat_lowest_to_excel(r"C:\Users\ytyan\Desktop\squat video\dadsquat.mp4")

 # def calculate_3d_angle_diff(p1, p2, p3, p4):
    #     xA, yA, zA = p1.x, 0 , p1.z
    #     xB, yB, zB = p2.x, 0 , p2.z
    #     xC, yC, zC = p3.x, 0 , p3.z
    #     xD, yD, zD = p4.x, 0 , p4.z
    #     v1 = np.array([xB - xA, yB - yA, zB - zA])
    #     v2 = np.array([xD - xC, yD - yC, zD - zC])
    #     #v3= np.array([xA, yA, zA]) + v2
    #     dot = np.dot(v1, v2)
    #     norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    #     return round(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))), 2)


    # def projected_angle_on_plane_4pt(A, B, C, D, P1, P2):
    #     """
    #     計算向量 P1->P2 投影到由四點(A,B,C,D)定義的平面後，
    #     與 Z 軸投影向量的夾角 (單位: 度)
    #     """
    #     # 轉 numpy
    #     A, B, C, D, P1, P2 = map(np.array, (A, B, C, D, P1, P2))
        
    #     # --- 1. 計算平面法向量
    #     n1 = np.cross(B - A, C - A)
    #     n2 = np.cross(C - A, D - A)
    #     n = (n1 + n2) / 2.0
    #     n = n / np.linalg.norm(n)
        
    #     # --- 2. 定義向量
    #     vec = P2 - P1
        
    #     # --- 3. 投影向量到平面
    #     forward_dir=np.array([0, 0, -1])
    #     vec_proj = vec - np.dot(vec, n) * n
    #     z_proj = forward_dir - np.dot(forward_dir, n) * n
        
    #     # --- 4. 單位化
    #     vec_proj = vec_proj / (np.linalg.norm(vec_proj) + 1e-9)
    #     z_proj = z_proj / (np.linalg.norm(z_proj) + 1e-9)
        
    #     # --- 5. 計算夾角
    #     cos_theta = np.dot(vec_proj, z_proj)
    #     cos_theta = np.clip(cos_theta, -1.0, 1.0)
    #     theta = np.degrees(np.arccos(cos_theta))
        
    #     # --- 6. 如果方向相反，用180-θ
    #     # if cos_theta < 0:
    #     #     theta = 180 - theta
        
    #     return theta
