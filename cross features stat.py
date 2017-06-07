import numpy as np
import pandas as pd
import f_pack

data = pd.read_csv(f_pack.file_train)
dfAd = pd.read_csv(f_pack.file_ad)
data = pd.merge(data, dfAd, how='left', on='creativeID')
'''
g1 = data.groupby(['positionID', 'connectionType']).apply(lambda x: np.mean(x["label"])).reset_index()
g1.columns = ['positionID', 'connectionType', 'mean_pos_conn']
g1.to_csv(f_pack.file_cf_pos_conn)

g2 = data.groupby(['positionID', 'advertiserID']).apply(lambda x: np.mean(x["label"])).reset_index()
g2.columns = ['positionID', 'advertiserID', 'mean_pos_adv']
g2.to_csv(f_pack.file_cf_pos_adv)
'''

g3 = data.groupby(['userID']).apply(lambda x: np.mean(x["label"])).reset_index()
g3.columns = ['userID', 'mean_userID']
g3.to_csv(f_pack.file_f_mean_userID)
