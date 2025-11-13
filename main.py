import matplotlib.pyplot as plt
import opendatasets as od
import numpy as np
import pandas as pd
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

fig,axs = plt.subplots(1,2, figsize=(15,7))
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
plt.show()