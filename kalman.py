import random
import numpy as np
import utils
from scipy.optimize import linear_sum_assignment

TERMINATE_SET = 5 # 设置轨迹终止计数，若连续5帧没有出现检测框（观测）则删除轨迹

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)  

class Kalman(object):
    def __init__(self, X):
        # 固定参数
        self.A = np.array([[1, 0, 0, 0, 1, 0],[0, 1, 0, 0, 0, 1],[0, 0, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]])  # 状态转移矩阵
        self.B = None  # 控制矩阵 无输入，设为0矩阵
        self.H = np.array([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0]])  # 观测矩阵
        self.Q = np.eye(self.A.shape[0]) * 0.1  # 过程噪声 << 观测噪声
        self.R = np.eye(self.H.shape[0])  # 观测噪声
        # 迭代参数
        self.X_posterior = X  # 定义为 [中心x,中心y,宽w,高h,dx,dy], 新出现的检测框直接作为最优后验状态
        self.P_posterior = np.eye(self.A.shape[0])  # 后验误差矩阵
        self.X_prior = None  # 先验状态
        self.P_prior = None  # 先验误差矩阵
        self.K = None  # 卡尔曼增益
        self.Z = None  # 观测, 定义为 [中心x,中心y,宽w,高h]
        # 起始和终止策略
        self.terminate_count = TERMINATE_SET
        self.track = []  # 记录当前轨迹[(p1_x,p1_y),()]
        self.track_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.__record_track()

    def predict(self):
        self.X_prior = self.A @ self.X_posterior
        self.P_prior = self.A @ self.P_posterior @ self.A.T + self.Q
    
    @staticmethod
    def associate_detections_to_predicts(detections, kalman_list, iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        kalman_list -> predict_list 预测  
        detections -> 观测  ndarray [c_x, c_y, w, h].T

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        # 将状态类进行转换便于统一匹配类型
        detections = [utils.mea2box(mea) for mea in detections]  # xyxy
        predict_list = list()  # [c_x, c_y, w, h].T  -> xyxy
        for kalman in kalman_list:
            state = kalman.X_prior  # t帧先验估计值
            predict_list.append(utils.mea2box(state[0:4]))

        if(len(kalman_list)==0):  # 对应初始化
            return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,1), dtype=int)

        iou_matrix = iou_batch(predict_list, detections)  # 行是预测，列是检测

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)  # 直接对应完全匹配
            else:
                x, y = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(x,y)))
        else:
            matched_indices = np.empty(shape=(0,2))

        unmatched_detections = []
        for d, _ in enumerate(detections):
            if(d not in matched_indices[:,1]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, _ in enumerate(predict_list):
            if(t not in matched_indices[:,0]):
                unmatched_trackers.append(t)

        #filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[1])  # 行是预测，列是检测
                unmatched_trackers.append(m[0])
            else:
                matches.append(m.reshape(1,2))

        if len(matches) == 0:
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, mea=None):
        status = True
        if mea is not None:  # 有匹配观测匹配上
            self.terminate_count = TERMINATE_SET  # 修改，如果有观测匹配上，更新终止计数
            self.Z = mea
            self.K = self.P_prior @ self.H.T @ np.linalg.inv(self.H @ self.P_prior @ self.H.T + self.R)# 计算卡尔曼增益
            self.X_posterior = self.X_prior + self.K @ (self.Z - self.H @ self.X_prior)  # 更新后验估计
            self.P_posterior = (np.eye(self.A.shape[0]) - self.K @ self.H) @ self.P_prior  # 更新后验误差矩阵
            status = True
        else:  # 无匹配观测匹配上
            if self.terminate_count == 1:
                status = False
            else:
                self.terminate_count -= 1
                self.X_posterior = self.X_prior
                self.P_posterior = self.P_prior
                status = True
        if status:
            self.__record_track()

        return status

    def __record_track(self):
        self.track.append([int(self.X_posterior[0]), int(self.X_posterior[1])])
