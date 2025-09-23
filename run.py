
def run_squat_analysis(video_path,squat_type):
    import cv2
    import mediapipe as mp
    import numpy as np
    import os
    import json
    import ffmpeg
    import requests
    from pose_analysis_module import analyze_single_squat
    from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
    from mediapipe.framework.formats import landmark_pb2
    import uuid
    api_url = "http://localhost/v1/chat-messages"  # Dify 本地部署 API 端點
    api_key = "app-lAcz4GKJGnPp1w7rrbmUl8jg"  # 你的金鑰

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    def analyze_issues(row,squat_type):
            issues = []
            # 設定深蹲標準角度範圍（膝蓋彎曲）
            if squat_type == "quarter":
                min_angle, max_angle = 105, 140
            elif squat_type == "half":
                min_angle, max_angle = 80, 105
            elif squat_type == "parallel":
                min_angle, max_angle = 50, 80
            elif squat_type == "full":
                min_angle, max_angle = 40, 55
            else:
                min_angle, max_angle = 60, 100  # 預設容許範圍（保守）

            # 使用你的欄位名稱對照實際資料
            right_angle = row['Right Depth Angle']
            left_angle = row['Left Depth Angle']

            # 判斷兩側是否在標準範圍內
            if not (min_angle <= right_angle <= max_angle) or not (min_angle <= left_angle <= max_angle):
                issues.append("深度不夠深")
        
            if row['diff_right'] > 12:
                issues.append("右腳膝蓋內夾")
            if row['diff_left'] > 12:
                issues.append("左腳膝蓋內夾")
            if row['diff_right'] < -7:
                issues.append("右腳膝蓋外開")
            if row['diff_left'] < -7:
                issues.append("左腳膝蓋外開")
            if abs(row['Right Depth Angle'] - row['Left Depth Angle']) > 10:
                issues.append("重心左右不平衡")
            return "、".join(issues) if issues else "✔"
    
    def filter_landmarks(landmarks, min_index=11):
        """過濾掉小於 min_index 的 landmark 點（如臉部 0~10）"""
        new_landmark_list = landmark_pb2.NormalizedLandmarkList()
        for i in range(len(landmarks.landmark)):
            if i >= min_index:
                new_landmark_list.landmark.append(landmarks.landmark[i])
        return new_landmark_list    
#------------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MIN_SQUAT_DURATION = 10
    FRAME_SMOOTH = 5
    HIP_THRESHOLD_RATIO = 0.05
    DEPTH_THRESHOLD_RATIO = 0.25

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # 設定影片輸出路徑
    os.makedirs(os.path.join(BASE_DIR, "static", "output"), exist_ok=True)
    raw_video_path = os.path.join(BASE_DIR, "static", "output", "annotated_output.mp4")
    html5_video_path = os.path.join(BASE_DIR, "static", "output", "annotated_output_html5.mp4")
    out = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    if not out.isOpened():
        print(" VideoWriter 初始化失敗，無法產生影片")
        return "影片寫入失敗", None
    else:
        print(" VideoWriter 成功建立，fps =", fps, " size =", frame_width, "x", frame_height)

    # 初始化變數
    history, all_frames_keypoints = [], []
    segment_frames = []
    report_table, error_image_paths = [], []
    records_list = []
    frame_idx = 0
    frame_skip = 2  # 每 2 幀處理 1 幀
    state = "idle"
    squat_id = 0
    fixed_leg_lengths = None
    min_hip_y = None
    max_hip_y = None
    HIP_Y_THRESHOLD = None
    MIN_SQUAT_DEPTH = None
    answer = None

    # EMA 平滑參數
    c_lmlist = []  # 收集 33 點 landmark 的歷史座標
    EMA_WINDOW = 5
    alpha = 2 / (EMA_WINDOW + 1)
    dived = 1 * (1 - (1 - alpha) ** EMA_WINDOW) / (1 - (1 - alpha))  # 等比數列和

    # 逐幀分析
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

        
        # 自定連線清單，排除臉部點（只留下主體骨架）
        BODY_CONNECTIONS = [
            (11, 12),  # 左右肩
            (11, 13), (13, 15),  # 左肩 → 左手肘 → 左手腕
            (12, 14), (14, 16),  # 右肩 → 右手肘 → 右手腕
            (15, 17), (17, 19), (15, 19), (15, 21),  # 左手掌
            (16, 18), (18, 20), (16, 22), (16, 20),  # 右手掌
            (23, 24),  # 左右髖
            (11, 23), (12, 24),  # 肩 → 髖
            (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # 左腿 → 腳
            (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)  # 右腿 → 腳
        ]
        if not (results and results.pose_landmarks and results.pose_world_landmarks):
            out.write(frame)
            frame_idx += 1
            continue

        # 收集 33 個點的原始 landmarks 資料（x, y, z, visibility）
        cur_lmlist = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        c_lmlist.append(cur_lmlist)

            # 滿足平滑條件才開始處理
        if len(c_lmlist) >= EMA_WINDOW:
            ema_list = []
            for i, q in zip(range(EMA_WINDOW), range(EMA_WINDOW - 1, -1, -1)):
                temp = [list(map(lambda x: x * (1 - alpha) ** q / dived, c_lmlist[i][j])) for j in range(33)]
                ema_list.append(temp)
            arr = np.array(ema_list)
            s_arr = np.sum(arr, axis=0).tolist()
            for i in range(33):
                results.pose_landmarks.landmark[i].x = s_arr[i][0]
                results.pose_landmarks.landmark[i].y = s_arr[i][1]
                results.pose_landmarks.landmark[i].z = s_arr[i][2]
                results.pose_landmarks.landmark[i].visibility = s_arr[i][3]
            del c_lmlist[0] # 移除最舊的一筆
                
        hip_y = ((results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y) / 2) * frame_height
        history.append((frame.copy(), hip_y, results.pose_landmarks.landmark, results.pose_world_landmarks.landmark))
        filtered_landmarks = filter_landmarks(results.pose_landmarks)
        BODY_CONNECTIONS_FILTERED = [(a-11, b-11) for (a, b) in BODY_CONNECTIONS]
        
        mp_drawing.draw_landmarks(
            frame,
            filtered_landmarks,
            BODY_CONNECTIONS_FILTERED,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,227,132), thickness=2, circle_radius=5),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(237,145,33), thickness=2)
        )
        

        # 儲存骨架資訊
        frame_keypoints = {
            'frame': frame_idx,
            'keypoints': {
                mp_pose.PoseLandmark(idx).name: {
                    'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility
                } for idx, lm in enumerate(results.pose_landmarks.landmark)
            }
        }
        all_frames_keypoints.append(frame_keypoints)

        # 動作分析狀態機
        if len(history) >= FRAME_SMOOTH:
            smoothed_hip_y = np.mean([h[1] for h in history[-FRAME_SMOOTH:]])
        else:
            smoothed_hip_y = hip_y
            
        # 臉部可見性檢查
        face_landmark_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        face_visible = all(results.pose_landmarks.landmark[i].visibility > 0.98 for i in face_landmark_ids)

        # 腿是否伸直（左腿為例）
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

        if fixed_leg_lengths is None and face_visible and slope_close:
            r_leg = np.linalg.norm([frame_width * (results.pose_landmarks.landmark[24].x - results.pose_landmarks.landmark[26].x),
                                    frame_height * (results.pose_landmarks.landmark[24].y - results.pose_landmarks.landmark[26].y)])
            l_leg = np.linalg.norm([frame_width * (results.pose_landmarks.landmark[25].x - results.pose_landmarks.landmark[23].x),
                                    frame_height * (results.pose_landmarks.landmark[25].y - results.pose_landmarks.landmark[23].y)])
            fixed_leg_lengths = (r_leg, l_leg)
            fixed_leg_length = (r_leg + l_leg) / 2
            HIP_Y_THRESHOLD = fixed_leg_length * HIP_THRESHOLD_RATIO
            MIN_SQUAT_DEPTH = fixed_leg_length * DEPTH_THRESHOLD_RATIO

        if state == "idle" and len(history) >= MIN_SQUAT_DURATION and HIP_Y_THRESHOLD is not None:
            prev_hip_y = np.mean([h[1] for h in history[-MIN_SQUAT_DURATION:-MIN_SQUAT_DURATION//2]])
            if face_visible and smoothed_hip_y - prev_hip_y > HIP_Y_THRESHOLD:
                state = "squatting"
                segment_frames = []
                min_hip_y = smoothed_hip_y
                max_hip_y = prev_hip_y
                

        elif state == "squatting":
            segment_frames.append((frame.copy(), hip_y, results.pose_landmarks.landmark, results.pose_world_landmarks.landmark))
            min_hip_y = min(min_hip_y, smoothed_hip_y)
            max_hip_y = max(max_hip_y, smoothed_hip_y)

            if len(history) >= MIN_SQUAT_DURATION:
                prev_hip_y = np.mean([h[1] for h in history[-MIN_SQUAT_DURATION:-MIN_SQUAT_DURATION//2]])
                if prev_hip_y - smoothed_hip_y > HIP_Y_THRESHOLD and (max_hip_y - min_hip_y) > MIN_SQUAT_DEPTH:
                    
                    squat_id += 1
                    lowest = max(segment_frames, key=lambda x: x[1])
                    
                    row = analyze_single_squat(lowest[2], lowest[3], lowest[0], squat_id, frame_width, frame_height, fixed_leg_lengths,squat_type)
                    #檢查是否有膝蓋錯誤
                    if abs(row['diff_right']) > 10 or abs(row['diff_left']) > 10:
                        error_frame_img = lowest[0].copy()  # 複製最低點影像
                        landmarks = lowest[2]               # mediapipe landmarks
                        frame_h, frame_w = error_frame_img.shape[:2]
                        '''lowest[0] → 該幀的 BGR 影像 (可直接用 cv2.imwrite)
                            lowest[2] → 該幀的 2D 關鍵點 (image coordinate normalized [0~1])
                            lowest[3] → 該幀的 3D 世界座標關鍵點 (含真實距離單位，通常是米，z 為深度)'''
                        
                        def draw_horizontal_arrow(img, x_center, y_center, direction="left", length=50, color=(0,0,255), thickness=5):
                            """
                            在 (x_center, y_center) 畫一個水平方向的箭頭
                            direction: "left" 或 "right"
                            """
                            if direction == "left":
                                start_pt = (x_center + length//2, y_center)
                                end_pt = (x_center - length//2, y_center)
                            else:
                                start_pt = (x_center - length//2, y_center)
                                end_pt = (x_center + length//2, y_center)
                            cv2.arrowedLine(img, start_pt, end_pt, color, thickness=thickness, tipLength=0.3)
                        
                        x1 = int(landmarks[26].x * frame_w)
                        y1 = int(landmarks[26].y * frame_h)
                        x2 = int(landmarks[28].x * frame_w)
                        y2 = int(landmarks[28].y * frame_h)
                        x_rcenter = (x1 + x2) // 2
                        y_rcenter = (y1 + y2) // 2 
                        x1 = int(landmarks[25].x * frame_w)
                        y1 = int(landmarks[25].y * frame_h)
                        x2 = int(landmarks[27].x * frame_w)
                        y2 = int(landmarks[27].y * frame_h)
                        x_lcenter = (x1 + x2) // 2
                        y_lcenter = (y1 + y2) // 2 
                        if row['diff_right'] < -15:
                            draw_horizontal_arrow(error_frame_img, x_rcenter, y_rcenter, direction="right", color=(255,227,132))
                        if row['diff_right'] > 15: #內夾 往外指
                            draw_horizontal_arrow(error_frame_img, x_rcenter, y_rcenter, direction="left", color=(255,227,132))          
                        
                        if row['diff_left'] > 15:
                            draw_horizontal_arrow(error_frame_img, x_lcenter, y_lcenter, direction="right", color=(255,227,132))
                        if row['diff_left'] < -15:
                            draw_horizontal_arrow(error_frame_img, x_lcenter, y_lcenter, direction="left", color=(255,227,132))
                        
                        
                        error_filename = f"error_squat_{squat_id}.jpg"
                        error_frame_path = os.path.join(BASE_DIR, "static", "output", error_filename)
                        cv2.imwrite(error_frame_path, error_frame_img)
                        error_image_paths.append({"img_path": f"output/{error_filename}", "squat_id": squat_id})
                    
                    report_table.append({
                        'index': squat_id,
                        'right_depth': f"{row['Right Depth Angle']:.2f}",
                        'left_depth': f"{row['Left Depth Angle']:.2f}",
                        'right_foot_angle_3d': f"{row['Right Foot Angle 3D (30→32)']:.2f}",
                        'left_foot_angle_3d': f"{row['Left Foot Angle 3D (29→31)']:.2f}",   
                        'right_diff': f"{row['diff_right']:.2f}",
                        'left_diff': f"{row['diff_left']:.2f}",
                        'issues': analyze_issues(row,squat_type),  # 下面會定義一個 helper function
                        'total_score': f"{row['Total Score']:.2f}"
                    })

                    
                    segment_frames = []
                    state = "idle"

        # 寫入影片每一幀（骨架疊合後）
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(" 所有幀寫入完成，準備轉檔")
    
    session_id = os.path.splitext(os.path.basename(video_path))[0]
    # 統計總分與平均分數
    total_score_sum = 0
    squat_count = len(report_table)

    if squat_count > 0:
        total_score_sum = sum(round(float(row['total_score']), 2) for row in report_table)
        average_score = total_score_sum / squat_count
    else:
        average_score = 0.0

    #收集所有深蹲數據report_table
    report_table = sorted(report_table, key=lambda x: int(x['index']))
    
    for row in report_table:
        row['index'] = int(row['index'])
    records_list = [
        {
            "index": row['index'], 
            "issues": row['issues'], 
            "total_score": float(row['total_score'])
        }
        for row in report_table
    ]
    
    query_content = f"請根據這個深蹲數據分析：{json.dumps(records_list, ensure_ascii=False)}"

    payload = {
        "inputs": {},  # inputs 留空或不傳遞
        "query": query_content,
        "response_mode": "blocking",
        "conversation_id": "", # 第一次請求不傳遞 conversation_id
        "user": session_id
    }

    llm_answer = "無法取得AI回覆"
    conversation_id = "" # 初始化對話 ID


    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        llm_answer = result.get("answer", llm_answer)
        conversation_id = result.get("conversation_id", conversation_id) # 擷取對話 ID
        print("LLM 診斷結果：", llm_answer)
        print("Conversation ID:", conversation_id)
    except Exception as e:
        print("Dify API 連線錯誤：", e)
    
                    
    # 影片轉為 HTML5 相容格式
    (
        ffmpeg
        .input(raw_video_path)
        .output(
            html5_video_path,
            vcodec='libx264',
            acodec='aac',
            strict='experimental',
            movflags='+faststart',
            pix_fmt='yuv420p'
        )
        .run(overwrite_output=True)
    )

    # 存 JSON
    json_output_path = os.path.join(BASE_DIR, "static", "output", "all_frames_keypoints.json")
    with open(json_output_path, "w") as f:
        json.dump(all_frames_keypoints, f, indent=2)

    return llm_answer, report_table, "output/annotated_output_html5.mp4", error_image_paths, round(average_score, 2), squat_count,conversation_id
