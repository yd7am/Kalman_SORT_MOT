# 卡尔曼滤波+匈牙利算法最大权匹配MOT

## 核心思想
第t帧图像卡尔曼先验预测估计与第t帧图像检测（观测）进行最大权值和匹配，对匹配的预测与观测进行卡尔曼更新步得到最优后验估计，对有预测没观测（漏检）的将预测作为最优后验估计，对有观测没预测（新目标）的将观测作为最优后验估计，即初始化 $ \hat{x}_0 $ 与 $ P_0 $
## Ref
[1] https://blog.csdn.net/ouok000/article/details/125578636 <br>
[2] https://github.com/liuchangji/kalman-filter-in-single-object-tracking <br>
[3] https://github.com/ZhangPHEngr/Kalman-in-MOT <br>
[4] https://blog.csdn.net/qq_42374559/article/details/96032672 <br>
[5] https://github.com/abewley/sort <br>
