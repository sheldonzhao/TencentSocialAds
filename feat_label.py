import pandas as pd
import f_pack


def stat(group):
    group = group.sort_values('minute')
    group = group.reset_index()
    group['h_potential'] = 0
    group['h_potential02'] = 0
    length = group.size / len(group.columns)
    if length < 2:
        return group

    minute = group['minute']

    if length == 2:
        if minute[1] - minute[0] <= 3 or (0 <= minute[1] <= 1 and 58 <= minute[0] < 60):
            group.loc[0, 'h_potential'] = 1
        return group

    count = 0
    for i in range(len(minute) - 1):
        if minute[i + 1] - minute[i] <= 3 or (0 <= minute[i + 1] <= 1 and 58 <= minute[i] < 60):
            count = 1
            group.loc[i, 'h_potential'] = 1
            if i > 0 and group.loc[i - 1, 'h_potential02'] == 1:
                group.loc[i, 'h_potential'] = 0
        if count == 1 and i < len(minute) - 2:
            if minute[i + 2] - minute[i + 1] <= 3 or (0 <= minute[i + 2] <= 1 and 58 <= minute[i + 1] < 60):
                group.loc[i, 'h_potential02'] = 1
        count = 0
    return group


data = pd.read_csv(f_pack.file_train)
print(data.shape)
data['day'] = data['clickTime'].map(lambda x: x // 10000)
days = data['day']
days = days.unique()
save_data = pd.DataFrame()
for day in days:
    test_data = data[data['day'] == day]
    print('test_data num is %d' % len(test_data))
    sub_data = data[data['day'] == day].copy()
    print('sub_data num is %d' % len(sub_data))
    sub_data['minute'] = sub_data['clickTime'] % 100
    sub_data = sub_data.groupby('userID', as_index=False).apply(stat).reset_index()
    sub_data = sub_data.drop(['level_0', 'level_1', 'index'], axis=1)
    print('sub_data num is %d' % len(sub_data))
    save_data = pd.concat([save_data, sub_data])
    print(save_data.shape)

save_data.to_csv(f_pack.file_train_s, index=False)
data = pd.read_csv(f_pack.file_train_s)
print(data.shape)
