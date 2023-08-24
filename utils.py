import numpy as np

def box2state(box):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return np.array([[center_x, center_y, w, h, 0, 0]]).T  # 定义为 [中心x,中心y,宽w,高h,dx,dy] 初始化速度为0


def state2box(state):
    center_x = state[0]
    center_y = state[1]
    w = state[2]
    h = state[3]
    return [int(i) for i in [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]]


def box2meas(box):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return np.array([[center_x, center_y, w, h]]).T  # 定义为 [中心x,中心y,宽w,高h]


def mea2box(mea):
    """[cx,cy,w,h] to [xyxy]"""
    center_x = mea[0]
    center_y = mea[1]
    w = mea[2]
    h = mea[3]
    return [int(i) for i in [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]]


def mea2state(mea):
    return np.row_stack((mea, np.zeros((2, 1))))


