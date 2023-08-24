import os
import cv2
import numpy as np
import utils
from kalman import Kalman

if __name__ == '__main__':
    imgs_folder = r'..\datasets\imgs'
    imgs_list = sorted([os.path.join(imgs_folder, f) for f in os.listdir(imgs_folder) if f.endswith('.png') or f.endswith('.jpg')])

    video_name = r'..\output\detect8.avi'
    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_name, fourcc, 5, (1024,640))

    # 检测框标签
    label_file_path = r'..\datasets\frame3start.txt'
    with open(label_file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    # 2. 逐帧滤波
    state_list = []  # 单帧目标状态信息，存kalman对象, 存储先验估计，第一帧设为空，0

    frame_counter = 3
    for filename in imgs_list:
        print(frame_counter)
        img = cv2.imread(filename, 0)  # 灰度图像
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # VideoWriter 默认使用BGR格式写入视频
        meas_list_frame = []
        for bbox_str in lines:
            bbox_data = np.array(bbox_str.split(","), dtype="int")
            if bbox_data[0] == frame_counter:
                meas_list_frame.append(bbox_data[1:5])
                cv2.rectangle(img, tuple(bbox_data[1:3]), tuple(bbox_data[3:5]), (0, 255, 0), 1,  cv2.LINE_AA)

        # -----Kalman Filter for multi-objects-------------------
        # 预测
        for target in state_list:
            target.predict()
        # 1.匹配
        mea_list = [utils.box2meas(mea) for mea in meas_list_frame]  # cx,cy,w,h
        matches, unmatched_detections, unmatched_trackers = Kalman.associate_detections_to_predicts(mea_list, state_list)


        # 1.匹配上的更新后验估计
        for index in matches:  # [state_index, mea_index]
            state_list[index[0]].update(mea_list[index[1]])  
        # 2.状态没匹配上的，更新一下，如果触发终止就删除
        state_del = list()
        for idx in unmatched_trackers:
            status = state_list[idx].update()
            if not status:
                state_del.append(idx)
        state_list = [state_list[i] for i in range(len(state_list)) if i not in state_del]
        # 3.观测没匹配上的，作为新生目标进行轨迹起始
        for idx in unmatched_detections:
            state_list.append(Kalman(utils.mea2state(mea_list[idx])))

        # 显示所有的state到图像上
        for kalman in state_list:
            pos = utils.state2box(kalman.X_posterior)
            cv2.rectangle(img, tuple(pos[:2]), tuple(pos[2:]), (0, 0, 255), 1,  cv2.LINE_AA)  # 红色框是最优估计

        # 绘制轨迹
        for kalman in state_list:
            tracks_list = kalman.track
            for idx in range(len(tracks_list) - 1):
                last_frame = tracks_list[idx]
                cur_frame = tracks_list[idx + 1]
                cv2.line(img, last_frame, cur_frame, kalman.track_color, 2)

        frame_counter += 1

        video_writer.write(img)

    video_writer.release()