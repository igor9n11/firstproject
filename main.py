import matplotlib.pyplot as plt
import opendatasets as od
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
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
plt.show()
plt.close()
figg, ax = plt.subplots(1,2, figsize=(13,6))
ax[0].boxplot([df['hours_studied'],df['sleep_hours']], tick_labels = ['Часы обучения', 'Часы сна'])
ax[0].set_title('Разброс времени сна/обучения')
ax[1].boxplot([df['attendance_percent'],df['previous_scores'], df['exam_score']],
                  tick_labels = ['посещаемость, %','Прошлые результаты','Текущий результат'])
ax[1].set_title('Разброс посещаемости, результатов экзаменов')
plt.show()
plt.close()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
data = df
std_data = pd.DataFrame(X_scaled, columns = X.columns)
## minmaxscaler
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)
X_train_mm, X_test_mm, y_train_mm, y_test_mm = train_test_split(
    X_mm, target, test_size=0.2, random_state=42)
rf_mm = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
rf_mm.fit(X_train_mm, y_train_mm)
rf_pred_mm = rf_mm.predict(X_test_mm)
r2_mm = r2_score(y_test_mm, rf_pred_mm)
mm_data = pd.DataFrame(X_mm, columns=X.columns)
figmm, axs = plt.subplots(3, 2, figsize=(12, 10))
axs[0,0].bar(data.index, data['hours_studied'], color='skyblue')
axs[0,0].set_title('Время учёбы')
axs[0,0].set_xlabel('Студенты')
axs[0,0].set_ylabel('Часы')
axs[0,1].bar(data.index, data['previous_scores'], color='skyblue')
axs[0,1].set_title('Предыдущие результаты (оригинал)')
axs[0,1].set_xlabel('Студенты')
axs[0,1].set_ylabel('Балл')
axs[1,0].bar(std_data.index, std_data['hours_studied'], color='red', alpha=0.7)
axs[1,0].set_title('Время учёбы (StandardScaler)')
axs[1,0].set_xlabel('Студенты')
axs[1,0].set_ylabel('Стандартизированное')
axs[1,1].bar(std_data.index, std_data['previous_scores'], color='red')
axs[1,1].set_title('Предыдущие результаты (StandardScaler)')
axs[1,1].set_xlabel('Студенты')
axs[1,1].set_ylabel('Стандартизированное')
axs[2,0].bar(mm_data.index, mm_data['hours_studied'], color='forestgreen')
axs[2,0].set_title('Время учёбы (MinMaxScaler)')
axs[2,0].set_xlabel('Студенты')
axs[2,0].set_ylabel('Значение')
axs[2,1].bar(mm_data.index, mm_data['previous_scores'], color='forestgreen')
axs[2,1].set_title('Предыдущие результаты (MinMaxScaler)')
axs[2,1].set_xlabel('Студенты')
axs[2,1].set_ylabel('Значение')
plt.suptitle('Сравнение методов предобработки')
plt.tight_layout()
plt.show()
plt.close()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size = 0.2, random_state = 42)
lr = LinearRegression()
lr.fit(X_train, y_train)
ylr_pred = lr.predict(X_test)
r2_1 = r2_score(y_test, ylr_pred)
r = Ridge()
r.fit(X_train, y_train)
yr_pred = r.predict(X_test)
r2_2 = r2_score(y_test, yr_pred)
rf = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
yrf_pred = rf.predict(X_test)
r2_3 = r2_score(y_test, yrf_pred)
################ Добавил 2 модели
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
ygb_pred = gb.predict(X_test)
r2_gb = r2_score(y_test, ygb_pred)
kn = KNeighborsRegressor(n_neighbors=5)
kn.fit(X_train, y_train)
ykn_pred = kn.predict(X_test)
r2_kn = r2_score(y_test, ykn_pred)
predictions = {
    'LR': ylr_pred,
    'Ridge': yr_pred,
    'RF': yrf_pred,
    'GB': ygb_pred,
    'KN': ykn_pred
}
results = []
for name, prediction in predictions.items():
    r2 = r2_score(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    results.append({
        'Модель': name,
        'r2': r2,
        'mae': mae,
        'mse': mse
    })
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('r2', ascending=False)
results_df.set_index('Модель',inplace=True)
print(results_df)

## 2 priznaka
f_names = X.columns
f_importance = pd.DataFrame({
    'feature': f_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
sorted_features = f_importance['feature'].tolist()
plt.figure(figsize=(10, 6))
plt.barh(f_importance['feature'], f_importance['importance'],
         color='skyblue', alpha=0.8, edgecolor='black')
plt.xlabel('Важность признака')
plt.title('Важность признаков', fontsize=14)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
plt.close()
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


lr_mm = LinearRegression()
lr_mm.fit(X_train_mm, y_train_mm)
lr_pred_mm = lr_mm.predict(X_test_mm)
r2_lr_mm = r2_score(y_test_mm, lr_pred_mm)
print(f"Результаты RandomForest с MinMaxScaler: {r2_mm}, Разница с StandartScaler: {abs(r2_mm-r2_3)}")
print(f"Результаты Линейной регрессии с MinMaxScaler: {r2_lr_mm}, Разница с StandartScaler: {abs(r2_mm-r2_1)}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, ylr_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color = 'red')
plt.xlabel("Реальный балл")
plt.ylabel("Предсказанный балл")
plt.title(f"Предсказания vs Реальность")
plt.grid(True, alpha=0.3)
plt.show()
plt.close()
## влияние гиперпараметров

ns1 = [10,55,100,300,555,1000]
rf_res = []
for n in ns1:
    rf_ns = RandomForestRegressor(n_estimators=n, max_depth=15, random_state=42, n_jobs=-1)
    rf_ns.fit(X_train, y_train)
    y_pred_ns = rf_ns.predict(X_test)
    r2 = r2_score(y_test, y_pred_ns)
    rf_res.append({'n_estimators': n, 'r2': r2})
rf_res_df = pd.DataFrame(rf_res)
print('Влияние n_estimators и max_depth для RandomForest:')
ns2 = [3,5,10,15,25,None]
rf_res1 = []
for n1 in ns2:
    rf_ns = RandomForestRegressor(n_estimators=500, max_depth=n1, random_state=42, n_jobs=-1)
    rf_ns.fit(X_train, y_train)
    y_pred_ns = rf_ns.predict(X_test)
    r2 = r2_score(y_test, y_pred_ns)
    rf_res1.append({'Depth': n1, 'r2': r2})
rf_res1_df = pd.DataFrame(rf_res1)
print('Влияние max_depth для RandomForest:')
vmeste_df = pd.concat([rf_res_df, rf_res1_df], axis=1)
print(vmeste_df)
ns3 = [1,3,5,10,20,30]
kn_res = []
for n2 in ns3:
    kn = KNeighborsRegressor(n_neighbors=n2)
    kn.fit(X_train, y_train)
    ykn_pred = kn.predict(X_test)
    r2 = r2_score(y_test, ykn_pred)
    kn_res.append({'Neighbors': n2, 'r2': r2})
kn_res_df = pd.DataFrame(kn_res)
print('Влияние n_neighbors для KNeighborsRegressor:')
print(kn_res_df)
input()
## neironka
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)
model = FCNN(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
train_losses = []
test_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            test_loss += criterion(out, yb).item()
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"epoch: {epoch+1}, train loss: {avg_train_loss}, test loss: {avg_test_loss}")
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        all_preds.extend(out.numpy())
        all_targets.extend(yb.numpy())
y_pred_nn = np.array(all_preds)
y_true_nn = np.array(all_targets)
r2_nn = r2_score(y_true_nn, y_pred_nn)
print(f"Точность r2 нейросети {r2_nn} ")
print(f"Разница с линейной регрессией {r2_nn - r2_1}")
print(f"Разница с RandomForest {r2_nn - r2_3}")