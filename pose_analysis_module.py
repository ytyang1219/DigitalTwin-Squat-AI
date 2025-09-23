# pose_analysis.py 負責運算與資料回傳 成功
import cv2
import numpy as np
import os
import math
from collections import namedtuple

Point2D = namedtuple("Point2D", ["x", "y"])
Point3D = namedtuple("Point3D", ["x", "y", "z"])

def analyze_single_squat(landmarks, world_landmarks, frame, count, frame_width, frame_height, fixed_leg_lengths, squat_type):
    id_list = list(range(23, 33))
    pixel = {i: Point2D(landmarks[i].x, landmarks[i].y) for i in id_list}
    world = {i: Point3D(world_landmarks[i].x, world_landmarks[i].y, world_landmarks[i].z) for i in id_list}
    
    def calculate_leg_open_angle(p1, p2, segment_length):
        delta_x = abs(p2.x - p1.x) * frame_width
        if segment_length == 0:
            return 0
        return np.degrees(np.arcsin(np.clip(delta_x / segment_length, -1.0, 1.0)))

    def leg_depth(p_upper, p_lower, leglength):
        delta_y = abs(p_lower.y - p_upper.y) * frame_height
        ratio = np.clip(delta_y / leglength, -1.0, 1.0)
        angle = np.degrees(np.arcsin(ratio))
        return round(90 + angle if p_lower.y - p_upper.y > 0 else 90 - angle, 2)
    
    def calculate_depth_angle(pA, pB, pC):
        v1 = np.array([pA.x - pB.x, pA.y - pB.y, pA.z - pB.z])
        v2 = np.array([pC.x - pB.x, pC.y - pB.y, pC.z - pB.z])
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return round(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))), 2)

    # def calculate_3d_foot_angle(p_base, p_tip):
    #     xA, yA, zA = p_base.x, p_tip.y, p_tip.z
    #     v1 = np.array([p_tip.x - p_base.x, p_tip.y - p_base.y, p_tip.z - p_base.z])
    #     v2 = np.array([xA - p_base.x, yA - p_base.y, zA - p_base.z])
    #     dot = np.dot(v1, v2)
    #     norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    #     return round(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))), 2)
    
    # def calculate_3d_foot_angle (p_heel, p_toe):
    #     v = np.array([p_toe.x - p_heel.x, p_toe.z - p_heel.z])
    #     # 跟 z 軸（鏡頭方向）或 x 軸（左右）算夾角，看你需求
    #     z_axis = np.array([0, 1])
    #     angle = np.arccos(np.clip(np.dot(v, z_axis) / (np.linalg.norm(v)), -1.0, 1.0))
    #     return np.degrees(angle)
    

    def log_map_to_angle(x):
        x_min = 37.686
        x_max = 82.368
        angle_max = 55

        # 設定 B 值（影響壓縮率）
        B = 0.1

        # A 根據最大值反推出來
        x_range = x_max - x_min
        A = angle_max / math.log(B * x_range + 1)

        if x <= x_min:
            return 0
        elif x >= x_max:
            return angle_max

        angle = A * math.log(B * (x - x_min) + 1)
        return angle
    def exp_map_to_angle(x):
        x_min = 37.686
        # 兩個參考點
        x1, angle1 = 72.599, 40
        x2, angle2 = 82.368, 60

        # 解 β
        β = math.log((angle2/angle1 + 1) / ( (angle1/angle1 + 1) )) / (x2 - x1)
        # 實際上 angle1 的公式要對應 (exp(β*(x1-x_min))-1)，這裡需要嚴格計算
        β = math.log((angle2/angle1 + 1)) / ( (x2 - x1) )

        # 再解 α
        α = angle1 / (math.exp(β * (x1 - x_min)) - 1)

        # 對輸入值做映射
        if x <= x_min:
            return 0
        elif x >= x2:
            return 60
        return α * (math.exp(β * (x - x_min)) - 1)

    def calculate_2d_angle(p_base, p_top):
        """
        計算從 base 到 top 的連線與水平軸的夾角（以 degree 表示）
        - 假設 y 軸向下（如 OpenCV 畫面）
        - 傾斜往上會是負角，往下為正角
        """
        dx = abs(p_top.x - p_base.x)
        dy = abs(p_top.y - p_base.y)

        angle_rad = np.arctan2(dx, dy)  # 可自動避免除以 0
        angle_deg = np.degrees(angle_rad)
        return round(angle_deg, 3)

   


    A_right = np.sqrt(fixed_leg_lengths[0] ** 2 - (pixel[26].y * frame_height - pixel[24].y * frame_height) ** 2)
    B_left = np.sqrt(fixed_leg_lengths[1] ** 2 - (pixel[25].y * frame_height - pixel[23].y * frame_height) ** 2)

    right_leg_angle = calculate_leg_open_angle(pixel[24], pixel[26], abs(A_right)) if A_right > 0 else 0
    left_leg_angle = calculate_leg_open_angle(pixel[25], pixel[23], abs(B_left)) if B_left > 0 else 0
    
    # right_foot_angle_3d = calculate_3d_foot_angle(world[30], world[32])
    # left_foot_angle_3d = calculate_3d_foot_angle(world[29], world[31])

    # diff_right = right_foot_angle_3d - right_leg_angle
    # diff_left = left_foot_angle_3d - left_leg_angle

    right_depth_angle = calculate_depth_angle(world[24], world[26], world[28])
    left_depth_angle = calculate_depth_angle(world[23], world[25], world[27])

    right_90_depth_angle = leg_depth(pixel[24], pixel[26], fixed_leg_lengths[0])
    left_90_depth_angle = leg_depth(pixel[23], pixel[25], fixed_leg_lengths[1])

    # angle_right = calculate_3d_angle(world[24], world[26], world[30], world[32])
    # angle_left = calculate_3d_angle(world[23], world[25], world[29], world[31])
 #--------------新方法2-------------------------------------------------
    # right_leg_angle = projected_angle_on_plane(world[30], world[32],world[29],world[24], world[26]) 
    # left_leg_angle = projected_angle_on_plane(world[30], world[32],world[29],world[23], world[25])

    # right_foot_angle_3d = projected_angle_on_plane(world[30], world[32],world[29],world[30], world[32])
    # left_foot_angle_3d = projected_angle_on_plane(world[30], world[32],world[29],world[29], world[31])

    # diff_right = right_foot_angle_3d - right_leg_angle
    # diff_left = left_foot_angle_3d - left_leg_angle
#---------------------------------------------------------------------------------------------------
#--------------新方法3----------------------------------

    # right_leg_angle = projected_angle_on_plane_4pt(
    #     world[29], world[30], world[31], world[32],
    #     world[24], world[26]
    # )

    # left_leg_angle = projected_angle_on_plane_4pt(
    #     world[29], world[30], world[31], world[32],
    #     world[23], world[25]
    # )

    # right_foot_angle_3d = projected_angle_on_plane_4pt(
    #     world[29], world[30], world[31], world[32],
    #     world[30], world[32]
    # )

    # left_foot_angle_3d = projected_angle_on_plane_4pt(
    #     world[29], world[30], world[31], world[32],
    #     world[29], world[31]
    # )

    # diff_right = right_foot_angle_3d - right_leg_angle
    # diff_left  = left_foot_angle_3d - left_leg_angle
# #-------------way4直接投影--------------
#     diff_right = calculate_3d_angle_diff(world[24], world[26], world[30], world[32])
#     diff_left = calculate_3d_angle_diff(world[23], world[25], world[29], world[31])
#----------------------------------
    right_foot_angle_3d = log_map_to_angle(calculate_2d_angle(pixel[30], pixel[32]))
    left_foot_angle_3d = log_map_to_angle(calculate_2d_angle(pixel[29], pixel[31]))
    
    diff_right = right_foot_angle_3d - right_leg_angle
    diff_left  = left_foot_angle_3d - left_leg_angle
    
    # 儲存圖片與回傳 DataFrame row
    os.makedirs("squat_outputs", exist_ok=True)
    filename = f"squat_outputs/lowest_pose_{count}.png"
    cv2.imwrite(filename, frame)
    
    #-------------------------計算分數---------------------------------------------------------------------------
    def calc_diff_score(diff):
    # 理想範圍為 -5 ~ 10，超過則扣分，最多扣 20
        if -5 <= diff <= 10:
            return 20
        A = abs(diff - 10) if diff > 10 else abs(-5 - diff)
        return max(0, 20 - min(A, 20))

    score_left_diff = calc_diff_score(diff_left)
    score_right_diff = calc_diff_score(diff_right)

    # 重心分數 30
    center_diff = abs(right_depth_angle - left_depth_angle)
    if center_diff < 6:
        score_center = 30
    else:
        B = min(center_diff - 6, 30)
        score_center = max(0, 30 - B)

    # 深度分數 30
    if squat_type == "quarter":
        min_angle, max_angle = 105, 140
    elif squat_type == "half":
        min_angle, max_angle = 80, 105
    elif squat_type == "parallel":
        min_angle, max_angle = 50, 75
    elif squat_type == "full":
        min_angle, max_angle = 20, 55
    else:
        min_angle, max_angle = 60, 100  # 預設

    depth = (right_depth_angle + left_depth_angle) / 2
    if min_angle <= depth <= max_angle:
        score_depth = 30
    elif depth < min_angle:
        C = min(min_angle - depth, 30)
        score_depth = max(0, 30 - C)
    else:
        D = min(depth - max_angle, 30)
        score_depth = max(0, 30 - D)

    total_score = score_left_diff + score_right_diff + score_center + score_depth

    

    return {
        "Index": count,
        "Right Leg Length (24-26)": abs(fixed_leg_lengths[0]),
        "Left Leg Length (23-25)": abs(fixed_leg_lengths[1]),
        "Right leg Angle": abs(right_leg_angle),
        "Left leg Angle": abs(left_leg_angle),
        "Right Foot Angle 3D (30→32)": right_foot_angle_3d,
        "Left Foot Angle 3D (29→31)": left_foot_angle_3d,
        "diff_right": diff_right,
        "diff_left": diff_left,
        "Right Depth Angle": right_depth_angle,
        "Left Depth Angle": left_depth_angle,
        "Right Depth Angle (90)": right_90_depth_angle,
        "Left Depth Angle (90)": left_90_depth_angle,
        # "Right Angle (24-26-30-32)": angle_right,
        # "Left Angle (23-25-29-31)": angle_left,
        "Score Left Diff": score_left_diff,
        "Score Right Diff": score_right_diff,
        "Score Center": score_center,
        "Score Depth": score_depth,
        "Total Score": total_score
    }
       


              