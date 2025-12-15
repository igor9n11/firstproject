import matplotlib.pyplot as plt
import opendatasets as od
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
pd.set_option('display.max_columns', None)
url = "https://www.kaggle.com/datasets/kainatjamil12/students-exams-score-analysis-dataset"
od.download(url)
df = pd.read_csv('./students-exams-score-analysis-dataset/student_exam_scores.csv')
print(df.head())
print(f'Количество строк (студентов): {df.shape[0]}')
print(f'Количество столбцов (признаков): {df.shape[1]}')
print(f'Столбцы: {df.columns}')
print(f'Сколько пропущенных значений:\n{df.isna().sum()}')
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f'Количество дупликатов:\n{duplicates}')
else:
    print('Дупликатов не найдено')
print(df.info())
print(df.describe())
X = df.drop(['student_id','exam_score'], axis=1)
target = df['exam_score']
#input()
fig,axs = plt.subplots(1,2, figsize=(17,7))
axs[0].hist(df['sleep_hours'], bins = 50, color='orchid', edgecolor='black', alpha=0.6)
axs[0].set_title('Распределение Часов сна студентов')
axs[0].set_xlabel('Длина сна, ч')
axs[0].set_ylabel('Кол-во студентов')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)
axs[1].hist(df['previous_scores'], bins = 50, color='skyblue', edgecolor='black', alpha=0.6)
axs[1].set_title('Распределение предыдущих результатов студентов')
axs[1].set_xlabel('Предыдущий результат')
axs[1].set_ylabel('Кол-во студентов')
axs[1].grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()
#plt.close()
#input()
figg, ax = plt.subplots(1,2, figsize=(13,6))
ax[0].boxplot([df['hours_studied'],df['sleep_hours']], tick_labels = ['Часы обучения', 'Часы сна'])
ax[0].set_title('Разброс времени сна/обучения')
ax[1].boxplot([df['attendance_percent'],df['previous_scores'], df['exam_score']],
                  tick_labels = ['посещаемость, %','Прошлые результаты','Текущий результат'])
ax[1].set_title('Разброс посещаемости, результатов экзаменов')
#plt.show()
#plt.close()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
data = df
std_data = pd.DataFrame(X_scaled, columns = X.columns)
#input()
figr, axs = plt.subplots(2,2, figsize=(14,9))
axs[0,0].bar(data.index, data['hours_studied']); axs[0,0].set_title('Время учёбы')
axs[0,0].set_xlabel('Кол-во студентов')
axs[0,0].set_ylabel('Часы')
axs[0,1].bar(data.index, data['previous_scores']); axs[0,1].set_title('Предыдущие результаты')
axs[0,1].set_xlabel('Кол-во студентов')
axs[0,1].set_ylabel('Балл')
axs[1,0].bar(std_data.index, std_data['hours_studied']); axs[1,0].set_title('Стандартизированное')
axs[1,0].set_xlabel('Кол-во студентов')
axs[1,1].bar(std_data.index, std_data['previous_scores']); axs[1,1].set_title('Стандартизированное')
axs[1,1].set_xlabel('Кол-во студентов')
#plt.show()
#plt.close()
#input()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size = 0.2, random_state = 42)
lr = LinearRegression()
lr.fit(X_train, y_train)
ylr_pred = lr.predict(X_test)
r2_1 = r2_score(y_test, ylr_pred)
print(f"Точность линейной регрессии: {r2_1}")
r = Ridge()
r.fit(X_train, y_train)
yr_pred = r.predict(X_test)
r2_2 = r2_score(y_test, yr_pred)
print(f"Точность линейной регрессии Ridge: {r2_2}")
rf = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
yrf_pred = rf.predict(X_test)
r2_3 = r2_score(y_test, yrf_pred)
print(f"Точность 'Случайного леса': {r2_3}")
#input()
f_names = X.columns
f_importance = pd.DataFrame({
    'feature': f_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(f_importance['feature'], f_importance['importance'],
         color='skyblue', alpha=0.8, edgecolor='black')
plt.xlabel('Важность признака')
plt.title('Важность признаков', fontsize=14)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.gca().invert_yaxis()
#plt.show()
#plt.close()
## 2 priznaka
sorted_features = f_importance['feature'].tolist()
f_top2 = sorted_features[:2]
X_top2 = df[f_top2]
scaler_2 = StandardScaler()
X_scaled_2 = scaler_2.fit_transform(X_top2)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_scaled_2, target, test_size=0.2, random_state=42)
rf_2 = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
rf_2.fit(X_train_2, y_train_2)
yrf_pred_2 = rf_2.predict(X_test_2)
r2_top2 = r2_score(y_test_2, yrf_pred_2)
lr_2 = LinearRegression()
lr_2.fit(X_train_2, y_train_2)
ylr_pred_2 = lr_2.predict(X_test_2)
r2_lr_top2 = r2_score(y_test_2, ylr_pred_2)
print(f"Результаты RandomForest с 2 признаками: {r2_top2} , Разница с 4 признаками {r2_3-r2_top2}")
print(f"Результаты Линейной регрессии с 2 признаками: {r2_lr_top2} , Разница с 4 признаками {r2_1-r2_lr_top2}")
#input()
plt.figure(figsize=(8, 6))
plt.scatter(y_test, ylr_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color = 'red')
plt.xlabel("Реальный балл")
plt.ylabel("Предсказанный балл")
plt.title(f"Предсказания vs Реальность")
plt.grid(True, alpha=0.3)
#plt.show()
#plt.close()

## minmaxscaler
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)
X_train_mm, X_test_mm, y_train_mm, y_test_mm = train_test_split(
    X_mm, target, test_size=0.2, random_state=42)
rf_mm = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
rf_mm.fit(X_train_mm, y_train_mm)
rf_pred_mm = rf_mm.predict(X_test_mm)
r2_mm = r2_score(y_test_mm, rf_pred_mm)

lr_mm = LinearRegression()
lr_mm.fit(X_train_mm, y_train_mm)
lr_pred_mm = lr_mm.predict(X_test_mm)
r2_lr_mm = r2_score(y_test_mm, lr_pred_mm)
print(f"Результаты RandomForest с MinMaxScaler: {r2_mm}, Разница с StandartScaler: {abs(r2_mm-r2_3)}")
print(f"Результаты Линейной регрессии с MinMaxScaler: {r2_lr_mm}, Разница с StandartScaler: {abs(r2_mm-r2_1)}")