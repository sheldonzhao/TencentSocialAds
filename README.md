# TencentSocialAds
腾讯社交广告高校算法大赛
objective: 预测移动广告被点击后激活的概率 [pCVR=P(conversion=1 | ad, user, context)]

Data Cleaning:

	1. 30th day is inaccuarate but valuable because the prediction problem is time-sensitive. So how to take advantage it? 由于转化回流时间有长有短，所以最后五天的label可能是不准确的，尤其是第30天。如果将第30天的数据全部删除，将会丢失大量有用的信息。如果全部保留，又引进了相当程度的噪声。而我们发现，转化回流时间是与APP ID有关的。于是我们统计了每个APP ID的平均转化回流时间，并且删除掉了第30天中平均转化回流时间偏长的数据。

Feature engineering：

	1. Feature types: raw features, statistic features, time-series features, cross features. 
	2. statistic features needs to do Bayesian smooth
	3. time-series features such as the number of installed app before clicktime, the number of installed app of the same type before clicktime
	4. How to select cross features? use xgb features importance -> run xgb again to get updated features importance
	5. How to code cross features? 1. Hash and onehotencoding  2. groupby -> transfer cross features to statistic features   
	
Data Set Construction:

	1. use data of 28,29th days to predict 31th day
	2. conversion ratio of 17th -20th seems unstable, we should remove it. 
	
Model and Training:

	1. Be careful of data leakage. 
	2. model ensemble should use different kind of models such as the combination of xgb ans LR. xgb and lightGBM are both tree-based model. So the result is not as good as imagination. 
	3. ensemble mothod: Weighted average, stacking, random seeds.
	4. final result can multiple a ratio to approach platform mean conversion ratio.


总结：前期一直沉迷于LR带来的成绩，没有想到后期随着特征的增加，LR无法很好的表达特征。没有及时用xgboost，导致成绩一直提升缓慢，也许错过了许多重要特征。对于稀疏矩阵的运用有待加强。刚开始xgb结果很差是因为代码错误（label列选错了,直接贴之前京东赛代码的恶果），还是要思考一下，下次碰到这种稀疏矩阵，怎么通过pandas能够很好的解决。

李沐指出，模型是使用离散特征还是连续特征，其实是一个“海量离散特征+简单模型” 同 “少量连续特征+复杂模型”的权衡。既可以离散化用线性模型，也可以用连续特征加深度学习。就看是喜欢折腾特征还是折腾模型了。通常来说，前者容易，而且可以n个人一起并行做，有成功经验；后者目前看很赞，能走多远还须拭目以待。
逻辑回归属于广义线性模型，表达能力受限；单变量离散化为N个后，每个变量有单独的权重，相当于为模型引入了非线性，能够提升模型表达能力，加大拟合。
离散化后可以进行特征交叉，由M+N个变量变为M*N个变量，进一步引入非线性，提升表达能力。


参考资料：
30th https://github.com/oooozhizhi/TencentSocialAdvertising-30th-solutions
26th https://jiayi797.github.io/categories/腾讯算法大赛-CVR预估/
23th https://github.com/BladeCoda/Tencent2017_Final_Coda_Allegro
20th https://github.com/shenweichen/Tencent_Social_Ads2017_Mobile_App_pCVR
14th https://github.com/freelzy/Tencent_Social_Ads
7th http://blog.csdn.net/ben3ben/article/details/74838338
