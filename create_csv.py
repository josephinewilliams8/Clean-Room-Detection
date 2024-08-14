import pandas as pd

df = pd.DataFrame({'date':[], 'time':[], 'color suit':[], 'machine':[]})
df.to_csv('Cleanroom Tracking.csv', sep=',', index=False, encoding='utf-8')