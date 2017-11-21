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

## 其他队伍分享的highlight
### rank 14th 队伍名：竟然有这些操作

Trick特征：
通过观察原始数据是不难发现的,有很多只有clickTime和label不一样的重复数据，按时间排序发现重复数据如果转化，label一般标在头或尾，少部分在中间，在训练集上出现的情况在测试集上也会出现，所以标记这些位置后onehot，让模型去学习，再就是时间差特征，关于trick我比赛分享的这篇文章有较详细的说明。比赛后期发现了几个和这个trick相类似的文章1和文章2，可以参考。

统计特征：
原始特征主要三大类：广告特征、用户特征、位置特征，通过交叉组合算统计构造特征，由于机器限制，统计特征主要使用了转化率，丢掉了点击次数和转化次数。初赛利用了7天滑窗构造，决赛采用了周冠军分享的clickTime之前所有天算统计。三组合特征也来自周冠军分享的下载行为和网络条件限制，以及用户属性对app需求挖掘出。贝叶斯平滑user相关的特征特别废时间，初赛做过根据点击次数阈值来操作转化率，效果和平滑差不多但是阈值选择不太准。

活跃数特征：
特征构造灵感来自这里,比如某个广告位的app种数。

均值特征：
比如点击某广告的用户平均年龄

平均回流时间特征：
利用回流时间方式不对的话很容易造成leackage，这里参考了官方群里的分享，计算了每个appID的平均回流时间，没有回流的app用其所在类的平均回流时间代替

用户流水和历史特征：
利用installed文件关联user和app获得历史统计特征，利用actions进行7天滑动窗口获得用户和app流水特征。

一些特征：

冷启动特征；
排序特征；
用户点击数和转化数过于稀疏，可以分别LenbelEncoder,然后拼接后LabelEncoder；
一天24小时以半小时为单位分箱；
连续特征离散化，如分箱离散化、参考决策树分裂点离散化、或用XGB叶子号离散化，再拼接原始离散特征送入FFM；
交叉验证方式构造统计特征防止leakage；
用户转化序列(比如:0010)；
删除最后几天中平均回流时间长的某些appID或者advertiserID数据(考虑回流时间和广告主或app相关)；
多窗口统计(1分钟内、1小时内、1天内...)，利用多窗口将样本转化为二维，送人CNN(很强的捕捉局部信息能力)，不采用pooling，采用drop-out等；

rank 20 队伍名：unknown
用户点击日志挖掘_2_1_gen_user_click_features.py
挖掘广告点击日志，从不同时间粒度（天，小时）和不同属性维度（点击的素材，广告，推广计划，广告主类型，广告位等）提取用户点击行为的统计特征。

用户安装日志挖掘 _2_2_gen_app_install_features.py
根据用户历史APP安装记录日志，分析用户的安装偏好和APP的流行趋势，结合APP安装时间的信息提取APP的时间维度的描述向量。这里最后只用了一种特征。

广告主转化回流上报机制分析_2_4_gen_tricks.py
不同的广告主具有不同的转化计算方式，如第一次点击算转化，最后一次点击算转化，安装时点击算转化，分析并构造相应描述特征，提升模型预测精度。

广告转化率特征提取_2_5_gen_smooth_cvr.py
构造转化率特征，使用全局和滑动窗口等方式计算单特征转化率，组合特征转化率，使用均值填充，层级填充，贝叶斯平滑，拉普拉斯平滑等方式对转化率进行修正。

广告描述向量特征提取_2_6_gen_ID_click_vectors.py
广告投放是有特定受众对象的，而特定的受众对象也可以描述广告的相关特性，使用不同的人口属性对广告ID和APPID进行向量表示，学习隐含的语义特征。
建模预测
使用多种模型进行训练，包括LightGBM,XGBoost,FFM和神经网络，最后进行多模型加权融合提高最终模型性能。

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
