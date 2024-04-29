"""
Module/Script Name: Rule-Based Climbing Detection using RTMO

Author: Evan Sean Sainani
Date: 29th April 2024
"""

import torch, torchvision
import numpy as np
import cv2
import time
import mmcv
from mmpose.apis import visualize
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmdeploy.apis import inference_model
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

global climbArea

# Define the original vertices of the polygon
climbArea = np.array([[1092, 396], [862, 697], [1072, 723], [1185, 418]], np.int32) # table coordinates from the left camera angle
# climbArea = np.array([[549,120], [547, 265], [613, 287], [587, 134]], np.int32) # table coordinates from the right camera angle
# climbArea = np.array([[1530,77], [1530, 427], [1690, 524] ,[1622, 128]], np.int32) # table coordinates from the right camera angle (alt)
# climbArea = np.array([[322, 318], [452, 386] ,[585, 271], [488, 234]], np.int32) # left meeting room table coordinates
# climbArea = np.array([[265, 247], [599, 463] ,[673, 303], [385, 201]], np.int32) # right meeting room table coordinates
# climbArea = np.array([[379, 267], [501, 425] ,[643, 387], [495, 259]], np.int32) # outside meeting room table coordinates
# climbArea = np.array([[423, 302], [501, 425] ,[643, 387], [473, 282]], np.int32) # waiting area coordinates
# climbArea = np.array([[367, 240], [540, 271] ,[521, 317], [318, 273]], np.int32) # smoking area coordinates
# climbArea = np.array([[464, 258], [540, 271] ,[521, 317], [448, 295]], np.int32) # grass area coordinates



'''

'''
def detectClimb(keypoint, keypoint_scores, kp_name, kpt_thr):
    """
    Given an ROI and a point p, find if p lies inside the ROI or not. The points lying on the border are considered inside.

    Args:
        keypoint (float): Keypoint of a predicted instance
        keypoint_scores (float): Keypoint confidence score 
        kp_name (string): Name of the keypoint
        kpt_thr (float): Threshold for the keypoint confidence score, keypoint confidences above this threshold will be checked if they lie in ROI

    Returns:
        Bool: If keypoint lies in the ROI
    """
    inside = False
    if keypoint_scores > kpt_thr:
        num_vertices = len(climbArea)
        x,y = keypoint[0], keypoint[1]
        # inside = False

        p1 = climbArea[0]

        for i in range(1, num_vertices + 1):
            p2 = climbArea[i % num_vertices]
            if y > min(p1[1] , p2[1]):
                if y <= max(p1[1] , p2[1]):
                    if x<= max(p1[0] , p2[0]):
                        x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                        if p1[0] == p2[0] or x <= x_intersection:
                            inside = not inside
            p1 = p2
        if inside:
            print(f'{kp_name} detected in ROI with confidence of {keypoint_scores}')
            cv2.putText(frame_vis, f'{kp_name} detected in ROI {keypoint_scores:.3f}%', (45, 410), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2)
    return inside


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start = time.time()

img = '/root/workspace/mmdeploy/mnt/videos/fight.mp4'

# RTMO TENSORRT
deploy_cfg = '/root/workspace/mmdeploy/configs/mmpose/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640.py'
model_cfg = '/root/workspace/mmdeploy/rtmo-l_16xb16-600e_coco-640x640.py'
device = 'cuda:0'
backend_model = ['/root/workspace/mmdeploy/mmdeploy_models/mmpose/trt-rtmo/end2end.engine']

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = [640,640]

coco_keypoints = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle"
}


cap = cv2.VideoCapture(img)
videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('/root/workspace/mmdeploy/mnt/vis_result/rtmo_ort_fight.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (videoWidth,videoHeight))
count = 0
total_processing_time = 0
prev_time = time.time()
cv2.namedWindow("Human Pose Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Human Pose Estimation ", videoWidth, videoHeight)
climbing = None
climbFlag = "Not detected"
climbCount = 0
instanceIdx=0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    #curTime = time.time()

    t1 = time.time()
    model_inputs, _ = task_processor.create_input(frame, input_shape)
    with torch.no_grad():
        result = model.test_step(model_inputs)
        t2 = time.time()

    pred_instances = result[0].pred_instances
    keypoints = pred_instances.keypoints
    keypoint_scores = pred_instances.keypoint_scores
    metainfo = 'config/_base_/datasets/coco.py'

    climbing = False

    frame_result = visualize(frame, keypoints, keypoint_scores, metainfo=metainfo, show=False)
    frame_vis = mmcv.rgb2bgr(frame_result)

    # if any of the knees are in the ROI, check if any of the feet are in the ROI. if true then climbing is detected. ** UPDATED RULE
    for k in range(len(keypoints)):
        if (detectClimb(keypoints[k][13],keypoint_scores[k][13],"Left knee", 0.98) or detectClimb(keypoints[k][14], keypoint_scores[k][14], "Right knee", 0.98)):
                if (detectClimb(keypoints[k][15],keypoint_scores[k][15], "Left foot", 0.975) or detectClimb(keypoints[k][16], keypoint_scores[k][16], "Right foot",0.975)):
                    climbing = True
                    instanceIdx = k
                    break

    # if any of the knees or feet are in the ROI, determine as a climb ** INITIAL RULE
    # for k in range(len(keypoints)):
    #     for i in idx:         
    #         climbing = detectClimb(keypoints[k][i],keypoint_scores[k][i], coco_keypoints[i], 0)
    #         if climbing:
    #             instanceIdx = k
    #             break
    #     if climbing:
    #         break

    if climbing:
        climbFlag = "Detected!"
        climbCount +=1
        print(f'Climb count increased to {climbCount}')
        color = (0, 0, 255)
        # left angle
        # cv2.putText(frame_vis, f'LK {keypoint_scores[instanceIdx][13]:.3f}%', (1290, 102), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        # cv2.putText(frame_vis, f'RK {keypoint_scores[instanceIdx][14]:.3f}%', (1290, 181), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        # cv2.putText(frame_vis, f'LF {keypoint_scores[instanceIdx][15]:.3f}%', (1290, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        # cv2.putText(frame_vis, f'RF {keypoint_scores[instanceIdx][16]:.3f}%', (1290, 340), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        # alt right angle 
        # cv2.putText(frame_vis, f'LK {keypoint_scores[instanceIdx][13]:.3f}%', (70, 662), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)
        # cv2.putText(frame_vis, f'RK {keypoint_scores[instanceIdx][14]:.3f}%', (70, 741), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)
        # cv2.putText(frame_vis, f'LF {keypoint_scores[instanceIdx][15]:.3f}%', (70, 820), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)
        # cv2.putText(frame_vis, f'RF {keypoint_scores[instanceIdx][16]:.3f}%', (70, 900), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)
        cv2.putText(frame_vis, f'Climb Detected', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)

    else:
        climbFlag = "Not Detected"
        color = (0, 255, 0)


    #processing_time = time.time() - curTime
    # print(f"Processing Time: {t2-t1} seconds")

    count += 1
    total_processing_time += (t2-t1)

   # Draw the polygon on the image
    cv2.polylines(frame_vis, [climbArea], isClosed=True, color=(255, 255, 255), thickness=1)
    # Display frame with FPS 
    fps = 1 / (time.time() - prev_time)
    # print(f"FPS: {fps}")
    # cv2.putText(frame_vis, f'FPS: {fps}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)

    # out.write(frame_vis)
    cv2.imshow("Human Pose Estimation", frame_vis)

    prev_time = time.time()

    if cv2.waitKey(10) & 0xFF == ord('x'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()


average_fps = count / total_processing_time

print(f'Average FPS: {average_fps:.2f}')
print(f'Climb Count: {climbCount}')
