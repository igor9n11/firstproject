import matplotlib.pyplot as plt
import opendatasets as od
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

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
fig,axs = plt.subplots(1,2, figsize=(13,7))
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
numbers_cols = df.select_dtypes(include=[np.number]).columns.tolist()
students_cols = df.select_dtypes(include=['object']).columns.tolist()
print(numbers_cols)
figg, ax = plt.subplots(1,2, figsize=(13,6))
ax[0].boxplot([df['hours_studied'],df['sleep_hours']], tick_labels = ['Часы обучения', 'Часы сна'])
ax[0].set_title('Разброс времени сна/обучения')
ax[1].boxplot([df['attendance_percent'],df['previous_scores'], df['exam_score']],
                  tick_labels = ['посещаемость, %','Прошлые результаты','Текущий результат'])
ax[1].set_title('Разброс посещаемости, результатов экзаменов')
#plt.show()
#plt.close()
print(df['exam_score'].max()<df['previous_scores'].mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
data = df
std_data = pd.DataFrame(X_scaled, columns = X.columns)
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size = 0.2, random_state = 42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
ylr_pred = lr.predict(X_test)
r2 = r2_score(y_test, ylr_pred)
print(ylr_pred)
print(r2)