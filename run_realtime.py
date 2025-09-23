# run_realtime.py
import threading

# 用於即時 SSE 推送分析、前端同步表格
report_table = []
table_lock = threading.Lock()

def gen_video_stream(squat_type):
    import cv2
    import mediapipe as mp
    import numpy as np
    import pyttsx3
    from pose_analysis_module import analyze_single_squat
    from mediapipe.framework.formats import landmark_pb2

    # ----- 參數與初始化 -----
    MIN_SQUAT_DURATION = 10
    FRAME_SMOOTH = 5
    HIP_THRESHOLD_RATIO = 0.05
    DEPTH_THRESHOLD_RATIO = 0.25

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ----- 主態與記錄 -----
    history = []
    segment = []
    state = "wait_start_tpose"
    phase = "idle"
    squat_id = 0
    fixed_leg_lengths = None
    min_hip_y = None
    max_hip_y = None
    frame_idx = 0

    engine = pyttsx3.init()

    # EMA 平滑處理參數
    c_lmlist = []  # 收集 landmark list（每幀的 33 點）
    EMA_WINDOW = 30
    alpha = 2 / (EMA_WINDOW + 1)
    dived = 1 * (1 - (1 - alpha) ** EMA_WINDOW) / (1 - (1 - alpha))

    def speak(text):
        engine.say(text)
        engine.runAndWait()

    def filter_landmarks(landmarks, min_index=11):
        new_landmark_list = landmark_pb2.NormalizedLandmarkList()
        for i in range(len(landmarks.landmark)):
            if i >= min_index:
                new_landmark_list.landmark.append(landmarks.landmark[i])
        return new_landmark_list

    def analyze_issues(row, squat_type):
        issues = []
        if squat_type == "quarter":
            min_angle, max_angle = 105, 140
        elif squat_type == "half":
            min_angle, max_angle = 80, 105
        elif squat_type == "parallel":
            min_angle, max_angle = 50, 80
        elif squat_type == "full":
            min_angle, max_angle = 40, 55
        else:
            min_angle, max_angle = 60, 100

        right_angle = row['Right Depth Angle']
        left_angle = row['Left Depth Angle']
        if not (min_angle <= right_angle <= max_angle) or not (min_angle <= left_angle <= max_angle):
            issues.append("深度不夠深")
        if row['diff_right'] > 15:
            issues.append("右腳膝蓋內夾")
        if row['diff_left'] > 15:
            issues.append("左腳膝蓋內夾")
        if row['diff_right'] < -15:
            issues.append("右腳膝蓋外開")
        if row['diff_left'] < -15:
            issues.append("左腳膝蓋外開")
        if abs(row['Right Depth Angle'] - row['Left Depth Angle']) > 10:
            issues.append("重心左右不平衡")
        return "、".join(issues) if issues else "✔"

    def is_y_close(landmarks, a, b, tol=0.05):
        """判斷兩點在 Y 軸方向是否接近"""
        return abs(landmarks[a].y - landmarks[b].y) < tol

    def check_tpose(landmarks):
        """檢查手部關鍵點是否構成 T-pose"""
        arm_pts = [11, 12, 13, 14, 15, 16]  # 左右肩、手肘、手腕

        # 可見度過低則直接不成立
        if not all(landmarks[i].visibility > 0.6 for i in arm_pts):
            return False

        # 左右手是否平行張開
        return (
            is_y_close(landmarks, 11, 13) and is_y_close(landmarks, 13, 15)
            and is_y_close(landmarks, 12, 14) and is_y_close(landmarks, 14, 16)
        )

    def face_visible(landmarks, threshold=0.8):
        """檢查臉部 11 個點是否 visibility > threshold"""
        face_pts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return all(landmarks[i].visibility > threshold for i in face_pts)

    
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        tpose_counter = 0
        is_tpose = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (520, 300))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                # 儲存原始 landmarks（x, y, z, visibility）
                cur_lmlist = [
                    [lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark
                ]
                c_lmlist.append(cur_lmlist)

                # 若達到平滑條件，進行指數平滑
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

                    # 將平滑結果覆蓋進 results.pose_landmarks
                    for i in range(33):
                        results.pose_landmarks.landmark[i].x = s_arr[i][0]
                        results.pose_landmarks.landmark[i].y = s_arr[i][1]
                        results.pose_landmarks.landmark[i].z = s_arr[i][2]
                        results.pose_landmarks.landmark[i].visibility = s_arr[i][3]

                    # 移除最舊一筆，維持視窗大小
                    del c_lmlist[0]

                
                
                landmarks = results.pose_landmarks.landmark
                is_tpose = check_tpose(landmarks)

                if is_tpose:
                    tpose_counter += 1
                else:
                    tpose_counter = 0

                # 加入臉部可見條件（可選擇是否開啟）
                if state == "wait_start_tpose" and tpose_counter >= 5 and face_visible(landmarks):
                    print("開始分析，T-pose 成功")
                    state = "analyzing"
                    tpose_counter = 0
                    continue

                elif state == "analyzing" and tpose_counter >= 5:
                    if squat_id == 0:
                        print("尚未完成任何有效深蹲！")
                        tpose_counter = 0
                        phase = "idle"
                        continue
                    else:
                        print("結束分析，T-pose 成功")
                        break
               

                if state == "analyzing":
                    hip_y = ((results.pose_landmarks.landmark[23].y +
                              results.pose_landmarks.landmark[24].y) / 2) * frame_height
                    history.append((frame.copy(), hip_y,
                                    results.pose_landmarks.landmark,
                                    results.pose_world_landmarks.landmark))

                    filtered_landmarks = filter_landmarks(results.pose_landmarks)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        filtered_landmarks,
                        [(a - 11, b - 11)
                         for a, b in mp.solutions.pose.POSE_CONNECTIONS if a >= 11 and b >= 11],
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )

                    if len(history) >= FRAME_SMOOTH:
                        smoothed_hip_y = np.mean([h[1] for h in history[-FRAME_SMOOTH:]])
                    else:
                        smoothed_hip_y = hip_y

                    if fixed_leg_lengths is None:
                        r_leg = np.linalg.norm([
                            frame_width * (
                                results.pose_landmarks.landmark[24].x -
                                results.pose_landmarks.landmark[26].x),
                            frame_height * (
                                results.pose_landmarks.landmark[24].y -
                                results.pose_landmarks.landmark[26].y)
                        ])
                        l_leg = np.linalg.norm([
                            frame_width * (
                                results.pose_landmarks.landmark[25].x -
                                results.pose_landmarks.landmark[23].x),
                            frame_height * (
                                results.pose_landmarks.landmark[25].y -
                                results.pose_landmarks.landmark[23].y)
                        ])
                        fixed_leg_lengths = (r_leg, l_leg)
                        fixed_leg_length = (r_leg + l_leg) / 2
                        HIP_Y_THRESHOLD = fixed_leg_length * HIP_THRESHOLD_RATIO
                        MIN_SQUAT_DEPTH = fixed_leg_length * DEPTH_THRESHOLD_RATIO

                    if phase == "idle" and len(history) >= MIN_SQUAT_DURATION:
                        prev_hip_y = np.mean([h[1] for h in history[-MIN_SQUAT_DURATION:-MIN_SQUAT_DURATION // 2]])
                        if smoothed_hip_y - prev_hip_y > HIP_Y_THRESHOLD:
                            phase = "squatting"
                            segment = []
                            min_hip_y = smoothed_hip_y
                            max_hip_y = smoothed_hip_y

                    elif phase == "squatting":
                        segment.append((frame.copy(), hip_y,
                                        results.pose_landmarks.landmark,
                                        results.pose_world_landmarks.landmark))
                        min_hip_y = min(min_hip_y, smoothed_hip_y)
                        max_hip_y = max(max_hip_y, smoothed_hip_y)

                        if len(history) >= MIN_SQUAT_DURATION:
                            prev_hip_y = np.mean([h[1] for h in history[-MIN_SQUAT_DURATION:-MIN_SQUAT_DURATION // 2]])
                            if prev_hip_y - smoothed_hip_y > HIP_Y_THRESHOLD and (max_hip_y - min_hip_y) > MIN_SQUAT_DEPTH:
                                lowest = max(segment, key=lambda x: x[1])
                                squat_id += 1
                                print(f"[DEBUG] state: {state}, phase: {phase}, tpose_counter: {tpose_counter}, squat_id: {squat_id}")

                                row = analyze_single_squat(
                                    lowest[2], lowest[3], lowest[0],
                                    squat_id, frame_width, frame_height,
                                    fixed_leg_lengths, squat_type
                                )
                                issue_text = analyze_issues(row, squat_type)
                                # 寫入全域 report_table，thread-safe
                                with table_lock:
                                    report_table.append({
                                        'index': squat_id,
                                        'right_depth': f"{row['Right Depth Angle']:.2f}",
                                        'left_depth': f"{row['Left Depth Angle']:.2f}",
                                        'right_diff': f"{row['diff_right']:.2f}",
                                        'left_diff': f"{row['diff_left']:.2f}",
                                        'issues': issue_text,
                                        'total_score': f"{row['Total Score']:.2f}"
                                    })
                                speak(issue_text)
                                phase = "idle"

            # 取代 cv2.imshow，將 frame yield 給 Flask /video_feed
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            frame_idx += 1

    cap.release()

def get_latest_analysis_info():
    # 給 Flask 用於 SSE（分析流），回傳累積的完整報表
    with table_lock:
        return report_table.copy()

# if __name__ == "__main__":
#     # 直接執行時，啟動攝影機並顯示即時分析畫面
#     squat_type = "half"  # 或根據需要改成 "quarter", "parallel", "full"
#     stream = gen_video_stream(squat_type)
#     import cv2
#     import numpy as np
# for jpeg_bytes in stream:
#     # 只取出 JPEG 內容
#     if b'\r\n\r\n' in jpeg_bytes:
#         jpeg_content = jpeg_bytes.split(b'\r\n\r\n', 1)[1]
#         # 去掉結尾的 b'\r\n'（如果有）
#         if jpeg_content.endswith(b'\r\n'):
#             jpeg_content = jpeg_content[:-2]
#         arr = np.frombuffer(jpeg_content, dtype=np.uint8)
#         frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#         if frame is not None:
#             cv2.imshow("Squat Realtime", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
# cv2.destroyAllWindows()