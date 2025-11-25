import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder
import numpy as np

df1 = pd.read_csv("airline_1.csv")

# na_df = df.isna().sum().reset_index()
# na_df.columns = ['Column', 'Missing Values']

# print('airline1.csv')
# print('Raw data')
# print(tabulate(na_df, headers='keys', tablefmt='psql'))
# print(f"\nSamples: {df.shape[0]}")

# drop columns
df1 = df1.drop(columns=['Title','Review Date','Reviews','Month Flown','Verified','Route', 'Type of Traveller'])

# make "recommend" columns Boolean-type
for i in range(int(df1.shape[0])):
    if df1.loc[i,'Recommended'] == 'yes':
        df1.loc[i,'Recommended'] = True
    elif df1.loc[i,'Recommended'] == 'no':
        df1.loc[i,'Recommended'] = False

#find type of class and ordinal encoding
traveller_type = []
for i in range(int(df1.shape[0])):
    if df1.iloc[i,df1.columns.get_loc('Class')] not in traveller_type:
        traveller_type.append(df1.loc[i,'Class'])
# print(traveller_type)

for i in range(int(df1.shape[0])):
    if df1.loc[i,'Class'] == 'Economy Class':
        df1.loc[i,'Class'] = 1
    elif df1.loc[i,'Class'] == 'Premium Economy':
        df1.loc[i,'Class'] = 2
    elif df1.loc[i,'Class'] == 'Business Class':
        df1.loc[i,'Class'] = 3
    elif df1.loc[i,'Class'] == 'First Class':
        df1.loc[i,'Class'] = 4

# Name, Airline id 형태로 라벨링
user_enc = LabelEncoder()
item_enc = LabelEncoder()

df1['user_id'] = user_enc.fit_transform(df1['Name'])
df1['item_id'] = item_enc.fit_transform(df1['Airline'])
df1 = df1.drop(columns=['Airline', 'Name'])

# 9. 컬럼 순서 재정렬
#    1) user_id, item_id 맨 앞으로
#    2) 'Overall Rating'은 맨 마지막
cols_final = ['user_id', 'item_id']

# 나머지 컬럼들 중 'Overall Rating'을 제외하고 이어붙임
cols_middle = [c for c in df1.columns if c not in ['user_id', 'item_id', 'Overall Rating']]

# 마지막에 Overall Rating 붙이기
cols_final += cols_middle + ['Overall Rating']

df1 = df1[cols_final]

# df1.to_csv('preprocessed_1.csv')

# # df = pd.read_csv('preprocessed_1.csv')
# na_df = df1.isna().sum().reset_index()
# na_df.columns = ['Column', 'Missing Values']

# print('airline1.csv')
# print('Preprocessed')
# print(tabulate(na_df, headers='keys', tablefmt='psql'))
# print(f"\nSamples: {df1.shape[0]}")




# df2 = pd.read_csv("airline_2.csv")
# # na_df = df.isna().sum().reset_index()
# # na_df.columns = ['Column', 'Missing Values']

# # print('airline2.csv')
# # print('Raw data')
# # print(tabulate(na_df, headers='keys', tablefmt='psql'))
# # print(f"\nSamples: {df.shape[0]}")


# # drop columns
# df2 = df2.dropna(subset=['overall_rating', 'cabin_flown'])
# df2 = df2.drop(columns=['link','title','author_country','date','content','aircraft','wifi_connectivity_rating','route', 'ground_service_rating', 'type_traveller'])


# # make "recommend" columns to Boolean-type
# for i in range(int(df2.shape[0])):
#     if df2.iloc[i,df2.columns.get_loc('recommended')] == 1:
#         df2.iloc[i,df2.columns.get_loc('recommended')] = True
#     elif df2.iloc[i,df2.columns.get_loc('recommended')] == 0:
#         df2.iloc[i,df2.columns.get_loc('recommended')] = False


# #find type of class and ordinal encoding
# traveller_type = []
# for i in range(int(df2.shape[0])):
#     if df2.iloc[i,df2.columns.get_loc('cabin_flown')] not in traveller_type:
#         traveller_type.append(df2.iloc[i,df2.columns.get_loc('cabin_flown')])
# # print(traveller_type)

# for i in range(int(df2.shape[0])):
#     if df2.iloc[i,df2.columns.get_loc('cabin_flown')] == 'Economy':
#         df2.iloc[i,df2.columns.get_loc('cabin_flown')] = 1
#     elif df2.iloc[i,df2.columns.get_loc('cabin_flown')] == 'Premium Economy':
#         df2.iloc[i,df2.columns.get_loc('cabin_flown')] = 2
#     elif df2.iloc[i,df2.columns.get_loc('cabin_flown')] == 'Business Class':
#         df2.iloc[i,df2.columns.get_loc('cabin_flown')] = 3
#     elif df2.iloc[i,df2.columns.get_loc('cabin_flown')] == 'First Class':
#         df2.iloc[i,df2.columns.get_loc('cabin_flown')] = 4

# cols = [
#     'seat_comfort_rating',
#     'cabin_staff_rating',
#     'food_beverages_rating',
#     'inflight_entertainment_rating',
#     'value_money_rating'
# ]
# dic = {
#     'seat_comfort_rating': 3,
#     'cabin_staff_rating': 4,
#     'food_beverages_rating': 3,
#     'inflight_entertainment_rating': 2,
#     'value_money_rating': 4
# }
# for col in cols:
#     df2[col] = df2[col].fillna(dic[col])

# # convert airline_name to Cammel Case
# df2['airline_name'] = (
#     df2['airline_name']
#     .str.replace('-', ' ', regex=False)        # '-' → ' '
#     .str.title()                               # 각 단어의 첫 글자만 대문자
#     .str.strip()                               # 양쪽 공백 제거 (안정성)
# )

# # 컬럼명 변경 (Preprocessed_1과 동일한 형태로)
# rename_map = {
#     'airline_name': 'Airline',
#     'author': 'Name',
#     'cabin_flown': 'Class',
#     'seat_comfort_rating': 'Seat Comfort',
#     'cabin_staff_rating': 'Staff Service',
#     'food_beverages_rating': 'Food & Beverages',
#     'inflight_entertainment_rating': 'Inflight Entertainment',
#     'value_money_rating': 'Value For Money',
#     'overall_rating': 'Overall Rating',
#     'recommended': 'Recommended'
# }

# df2 = df2.rename(columns=rename_map)

# # factor들의 값을 실수형에서 정수형으로 변경
# df2['Seat Comfort'] = df2['Seat Comfort'].astype(int)
# df2['Staff Service'] = df2['Staff Service'].astype(int)
# df2['Food & Beverages'] = df2['Food & Beverages'].astype(int)
# df2['Inflight Entertainment'] = df2['Inflight Entertainment'].astype(int)
# df2['Value For Money'] = df2['Value For Money'].astype(int)
# df2['Overall Rating'] = df2['Overall Rating'].astype(int)

# # Name, Airline id 형태로 라벨링
# df2['user_id'] = user_enc.fit_transform(df2['Name'])
# df2['item_id'] = item_enc.fit_transform(df2['Airline'])
# df2 = df2.drop(columns=['Airline', 'Name'])

# # 9. 컬럼 순서 재정렬
# #    1) user_id, item_id 맨 앞으로
# #    2) 'Overall Rating'은 맨 마지막
# cols_final = ['user_id', 'item_id']

# # 나머지 컬럼들 중 'Overall Rating'을 제외하고 이어붙임
# cols_middle = [c for c in df2.columns if c not in ['user_id', 'item_id', 'Overall Rating']]

# # 마지막에 Overall Rating 붙이기
# cols_final += cols_middle + ['Overall Rating']

# df2 = df2[cols_final]

# df2.to_csv('preprocessed_2.csv')

# # # df = pd.read_csv('preprocessed_2.csv')
# # na_df = df2.isna().sum().reset_index()
# # na_df.columns = ['Column', 'Missing Values']

# # print('airline2.csv')
# # print('Preprocessed')
# # print(tabulate(na_df, headers='keys', tablefmt='psql'))
# # print(f"\nSamples: {df2.shape[0]}")