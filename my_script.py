#import necessary libs
import time
import numpy as np
import pandas as pd

#input path with the filename
filename = input('File name:')

print('Execution started. Takes up to 137.40 sec on my PC')
start_time = time.time()

#import file and parse date into datetime
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df_logs = pd.read_csv(filename,parse_dates=['ts'], date_parser = dateparse, index_col=None)

#add necessary columns for aggregation
df_logs['highly_active'] = 1
df_logs['day_of_month'] = df_logs['ts'].dt.day
df_logs['day_of_week'] = df_logs['ts'].dt.dayofweek
df_logs['hour'] = df_logs['ts'].dt.hour
df_logs['weekday_biz'] = (df_logs['hour'].isin(range(9,19))) & (df_logs['day_of_week'].isin(range(1,6)))

#aggregation
#activity
a = df_logs[['uuid','highly_active']].groupby('uuid').agg('sum')
#weekday business hours
b = df_logs[['uuid','weekday_biz']].groupby('uuid').agg(lambda x:x.value_counts().index[0])
#my_feature
c = df_logs[['uuid','hashed_ip']].groupby('uuid').agg({'hashed_ip': pd.Series.nunique})
c = c.T.squeeze()
c = c.rename('my_feature')
#multiple_days
d = df_logs.groupby('uuid')['day_of_month'].nunique()
d = d.rename('multiple_days')

#build a new dataframe with aggregation information
output = pd.concat([a,d, b,c], axis=1).reset_index()

#adjust features
threshold = output['highly_active'].quantile(0.75)
output['highly_active'] = np.where(output['highly_active']>=threshold, True, False)
output['multiple_days'] = np.where(output['multiple_days']>1, True,False)
output['my_feature'] = np.where(output['my_feature']>1, True,False)

#export
output.to_csv('output.csv', sep='\t')

print("Success. It took %s sec" % (time.time() - start_time))

