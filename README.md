# Перестановки

Есть два метода генерации случайных перестановок длины 8 — правильный и творческий. Код этих методов приведён ниже.

```python
def RandomPermutation():
    perm = list(range(8))
    random.shuffle(perm)
    return perm

def StupidPermutation():
    partialSums = [0,1,8,35,111,285,
        628,1230,2191,3606,5546,8039,11056,14506,18242,  
        22078,25814,29264,32281,34774,36714,38129,39090,  
        39692,40035,40209,40285,40312,40319,40320]
    r = random.randint(0, partialSums[-1])
    numInv = 0
    while partialSums[numInv] < r:
        numInv += 1
    perm = list(range(8))
    for step in range(numInv):
        t1 = random.randint(0, 7)
        t2 = random.randint(0, 7)
        perm[t1], perm[t2] = perm[t2], perm[t1]
    return perm
```

В правильном методе генерируется случайная перестановка. В творческом сначала выбирается число $numInv$, соответствующее доле неправильно упорядоченных пар чисел в случайной перестановке. Массив $partialSums$ выбран таким образом, что распределение величины $numInv$ получается правильным.

Дальше допущена ошибка, в результате которой не все перестановки будут получаться с одинаковыми вероятностями. Получив достаточно большое количество перестановок, сгенерированных одним из методов, можно угадать, что это был за метод.

В этой задаче по набору из 1000 перестановок необходимо определить, каким методом он был получен. Дано $n$ таких наборов, нужно отсортировать их так, чтобы сначала шли хорошие наборы, а потом творческие.

Решение засчитывается, если среди всех пар наборов (хороший, творческий), хотя бы 98% идёт в правильном порядке

## Формат ввода
Файл `permutations.in`

В первой его строке указано одно число $n$ — количество наборов перестановок. В каждой из следующих 1000 строк указано по перестанове чисел от 0 до 7. Первая 1000 строк соответствет первому набору перестановок, вторая 1000 второму и т.д.

## Формат вывода
Выведите $n$ чисел — в $i$-ой строке должен быть указан номер набора перестановок (от 0 до $n−1$). Сначала должны идти номера хороших наборов, потом творческих.


```python
import random
from datetime import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from IPython import display
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib.itertools import product
from tqdm.notebook import trange
```


```python
pio.templates.default = "plotly_dark"
```

Объявляем функции


```python
def RandomPermutation():
    perm = list(range(8))
    random.shuffle(perm)
    return perm

def StupidPermutation():
    partialSums = [0,1,8,35,111,285,
        628,1230,2191,3606,5546,8039,11056,14506,18242,  
        22078,25814,29264,32281,34774,36714,38129,39090,  
        39692,40035,40209,40285,40312,40319,40320]
    r = random.randint(0, partialSums[-1])
    numInv = 0
    while partialSums[numInv] < r:
        numInv += 1
    perm = list(range(8))
    for step in range(numInv):
        t1 = random.randint(0, 7)
        t2 = random.randint(0, 7)
        perm[t1], perm[t2] = perm[t2], perm[t1]
    return perm
```

Давайте построим столбчатые диаграммы для анализа взаимосвязи между местом элемента в перестановке и его значением.


```python
N = 100_000

statRandomPermutation = [[0 for _ in range(8)] for _ in range(8)]

for _ in range(N):
    for i, p in enumerate(RandomPermutation()):
        statRandomPermutation[p][i] += 100 / N


statStupidPermutation = [[0 for _ in range(8)] for _ in range(8)]

for _ in range(N):
    for i, p in enumerate(StupidPermutation()):
        statStupidPermutation[p][i] += 100 / N
```


```python
fig = go.Figure()

for i in range(8):
    fig.add_trace(
        go.Bar(
            x=list(range(1, 9)),
            y=statRandomPermutation[i],
            name=f"Число {i}",
            text=list(map(lambda x: round(x, 2), statRandomPermutation[i])),
            textposition='inside',
        )
    )

fig.update_layout(xaxis_title="Место", yaxis_title="%",barmode='stack',  title="RandomPermutation")
fig.show()
```

![](https://github.com/Cyber-Zhaba/prediction-of-permutations/blob/master/img/fig1.svg)

```python
fig = go.Figure()

for i in range(8):
    fig.add_trace(
        go.Bar(
            x=list(range(1, 9)),
            y=statStupidPermutation[i],
            name=f"Число {i}",
            text=list(map(lambda x: round(x, 2), statStupidPermutation[i])),
            textposition='inside',
        )
    )

fig.update_layout(xaxis_title="Место", yaxis_title="%",barmode='stack', title="StupidPermutation")
fig.show()
```

![](https://github.com/Cyber-Zhaba/prediction-of-permutations/blob/master/img/fig2.svg)

В первой диаграмме не удалось обнаружить явных отклонений. Во второй же диаграмме наблюдается следующая закономерность: значение, соответствующее определённому номеру места, часто совпадает с этим номером. Например, на месте с номером 0 чаще всего встречается значение 0, на первом месте — число 1 и так далее.

Давайте попробуем решить задачу с помощью простых линейных регрессий


```python
N = 10_000
df = pd.DataFrame(np.zeros((N, 9)))

for i in range(N):
    if np.random.random() < 0.5:
        for j, p in enumerate(RandomPermutation()):
            df.loc[i, j] = p
        df.loc[i, 8] = 1
    else:
        for j, p in enumerate(StupidPermutation()):
            df.loc[i, j] = p
        df.loc[i, 8] = 0
        
y = df[8]
X = df.drop(columns=[8,])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
```


```python
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
accuracy_score(LogReg.predict(X_test), y_test)
```




    0.522




```python
SGDC = SGDClassifier()
SGDC.fit(X_train, y_train)
accuracy_score(SGDC.predict(X_test), y_test)
```




    0.4888




```python
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
accuracy_score(KNN.predict(X_test), y_test)
```




    0.5108




```python
GBC = GradientBoostingClassifier()
GBC.fit(X_train, y_train)
accuracy_score(GBC.predict(X_test), y_test)
```




    0.5312



Точность таких решений очень невысока, она сравнима с точностью случайного выбора. По условиям задачи, чтобы решение было засчитано, оно должно давать 98% верных ответов. Давайте попробуем решить задачу другим способом.

Сначала нужно написать функцию для создания входного тензора.


```python
def generateRandomData(sz=1000, _device='cpu'):
    if np.random.random() < 0.5:
        x = torch.Tensor([
            RandomPermutation()
            for _ in range(sz)
        ]).to(_device)
        y = torch.Tensor([sz]).to(_device)
        return x, y
    else:
        x = torch.Tensor([
            StupidPermutation()
            for _ in range(sz)
        ]).to(_device)
        y = torch.Tensor([0]).to(_device)
        return x, y
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Для начала создадим сеть, состоящую из трёх слоёв: входного, выходного и скрытого. Входной слой будет содержать 8 нейронов, выходной — 1 нейрон, а количество нейронов в скрытом слое может меняться.

Затем мы оптимизируем параметры сети, используя метод перебора гиперпараметров, чтобы найти наилучшее значение для скрытого слоя.


```python
class Net1(nn.Module):
    def __init__(self, midLayer):
        super().__init__()
        self.input = nn.Linear(8, midLayer)
        self.output = nn.Linear(midLayer, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.sigmoid(self.output(x))
        return x
```

Функция для обучения


```python
def train(epochs, learning_rate, batch_size, modelClass, model_args, trainDataFunc, prefix="", verbose=True):
    _model = modelClass(*model_args).to(device)
    optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    writer = SummaryWriter(f'runs/{prefix}BS {batch_size} LR {learning_rate} TE {epochs} {dt.now().timestamp()}')

    pbar = trange(epochs) if verbose else range(epochs)
    scores = []

    for epoch in pbar:
        x, y = trainDataFunc(batch_size, device)

        optimizer.zero_grad()
        out = _model(x).sum().unsqueeze(-1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if epochs <= 100 or round(epoch / epochs * 100) > round((epoch - 1) / epochs * 100):
            with torch.no_grad():
                length = 1000
                scores = []
                if trainDataFunc == generateRandomData:
                    cmp = length / 2
                else:
                    cmp = 0.5
                for _ in range(1000 if epoch == epochs - 1 else 175):
                    x, y = trainDataFunc(length, device)
                    if y[0] > 0:
                        scores.append(int(
                            _model(x).sum() > cmp
                        ))
                    else:
                        scores.append(int(
                            _model(x).sum() < cmp
                        ))
                    
            writer.add_scalar("Accuracy", sum(scores) / len(scores), epoch)

    writer.add_hparams({"LR": learning_rate, "BS": batch_size, "Mid layer": model_args[0]}, {"accuracy": sum(scores) / len(scores)})

    return _model
```

Перебор гиперпараметров


```python
ls_list = [0.1, 0.05, 0.01, 0.005]
bs_list = [250, 500, 1000, 2000, 4000]
mid_layer = [8, 16, 32, 64, 128]

for ls, bs, mid in product(ls_list, bs_list, mid_layer):
    train(1000, ls, bs, Net1, (mid,), generateRandomData, "HSearch", False)
```


Построим график с параллельными координатами


```python
df = pd.read_csv("hparams_table.csv")
df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LR</th>
      <th>BS</th>
      <th>Mid layer</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.005</td>
      <td>1000.0</td>
      <td>8.0</td>
      <td>0.497417</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.005</td>
      <td>1000.0</td>
      <td>16.0</td>
      <td>0.500500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005</td>
      <td>1000.0</td>
      <td>32.0</td>
      <td>0.501917</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.005</td>
      <td>1000.0</td>
      <td>64.0</td>
      <td>0.519000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005</td>
      <td>1000.0</td>
      <td>128.0</td>
      <td>0.509417</td>
    </tr>
  </tbody>
</table>





```python
fig = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=df["accuracy"],
            colorscale=[(0.00, "red"),   (0.33, "red"),
                        (0.33, "green"), (0.66, "green"),
                        (0.66, "blue"),  (1.00, "blue")], 
            showscale=True,
        ),
        dimensions=list([
            dict(label='LR', values=df["LR"]),
            dict(label='BS', values=df["BS"]),
            dict(label='Mid layer', values=df["Mid layer"]),
            dict(label='accuracy', values=df["accuracy"]),
        ])
    )
)

fig.update_layout(height=600)

fig.show()
```

![](https://github.com/Cyber-Zhaba/prediction-of-permutations/blob/master/img/fig3.svg)


```python
fig = go.Figure(data=
    go.Parcoords(
        line_color='blue',
        dimensions = list([
            dict(label = 'LR', values = df["LR"]),
            dict(label = 'BS', values = df["BS"]),
            dict(label = 'Mid layer', values = df["Mid layer"]),
            dict(label = 'accuracy', values = df["accuracy"], constraintrange=[0.518, 520]),
        ])
    )
)

fig.update_layout(height=600)

fig.show()
```

![](https://github.com/Cyber-Zhaba/prediction-of-permutations/blob/master/img/fig4.svg)

Оптимальными параметрами для модели являются:
- скорость обучения — 0,005;
- размер батча — 1000;
- количество нейронов в среднем слое — 64.

Обучим модель на протяжении 100 000 эпох. В конце каждой эпохи будем определять промежуточную точность модели (Accuracy).


```python
model_1 = train(100_000, 0.005, 1000, Net1, (64,), generateRandomData)
```

    
![svg](https://github.com/Cyber-Zhaba/prediction-of-permutations/blob/master/img/fig5.svg)
    



Несмотря на большое количество эпох обучения, результаты остаются недостаточно высокими. Итоговая точность после 100 000 эпох составляет всего лишь 0,64, что является неприемлемым.

Давайте попробуем решить задачу по-другому. Мы можем построить решение на основе корреляции между номером места и числом. Как видно из столбчатых диаграмм, у первой функции такая корреляция отсутствует, поэтому процент чисел, совпадающих с номером места, не превышает 13. В то же время, у второй функции такая закономерность присутствует, поэтому процент стабильно выше 14.

Давайте воспользуемся этой разницей при построении модели. Структура сети останется прежней, но входные данные изменятся. Теперь на вход будут подаваться не сами числа, а доля чисел, номера которых совпадают с номером места.

Создадим функцию для генерации таких данных.


```python
def generateFracData(sz=1000, _device='cpu'):
    if np.random.random() < 0.5:
        x_arr = np.zeros(8)
        for _ in range(sz):
            for i, p in enumerate(RandomPermutation()):
                if i == p:
                    x_arr[i] += 1 / sz
        
        y = torch.Tensor([1]).to(_device)

    else:
        x_arr = np.zeros(8)
        for _ in range(sz):
            for i, p in enumerate(StupidPermutation()):
                if i == p:
                    x_arr[i] += 1 / sz
        
        y = torch.Tensor([0]).to(_device)
        
    x = torch.Tensor(x_arr).to(_device)
    return x, y
```

Перебор гиперпараметров


```python
ls_list = [0.1, 0.01, 0.001]
mid_layer = [2, 4, 8, 16, 32, 64, 128, 256, 512]

for ls, mid in product(ls_list, mid_layer):
    train(1000, ls, 1000, Net1, (mid,), generateFracData, "HSearch", False)
```

```python
df = pd.read_csv("hparams_table_1.csv")
df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LR</th>
      <th>BS</th>
      <th>Mid layer</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.010</td>
      <td>1000.0</td>
      <td>16.0</td>
      <td>0.546667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.010</td>
      <td>1000.0</td>
      <td>16.0</td>
      <td>0.560000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.001</td>
      <td>1000.0</td>
      <td>2.0</td>
      <td>0.466667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.001</td>
      <td>1000.0</td>
      <td>4.0</td>
      <td>0.493333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.001</td>
      <td>1000.0</td>
      <td>8.0</td>
      <td>0.506667</td>
    </tr>
  </tbody>
</table>




```python
fig = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=df["accuracy"],
            colorscale=[(0.00, "red"),   (0.33, "red"),
                                                     (0.33, "green"), (0.66, "green"),
                                                     (0.66, "blue"),  (1.00, "blue")], 
            showscale=True,
        ),
        dimensions=list([
            dict(label='LR', values=df["LR"]),
            dict(label='BS', values=df["BS"]),
            dict(label='Mid layer', values=df["Mid layer"]),
            dict(label='accuracy', values=df["accuracy"]),
        ])
    )
)

fig.update_layout(height=600)
```

![](https://github.com/Cyber-Zhaba/prediction-of-permutations/blob/master/img/fig6.svg)

```python
fig = go.Figure(data=
    go.Parcoords(
        line_color='blue',
        dimensions = list([
            dict(label = 'LR', values = df["LR"]),
            dict(label = 'BS', values = df["BS"]),
            dict(label = 'Mid layer', values = df["Mid layer"]),
            dict(label = 'accuracy', values = df["accuracy"], constraintrange=[0.9, 1]),
        ])
    )
)

fig.update_layout(height=600)

fig.show()
```

![](https://github.com/Cyber-Zhaba/prediction-of-permutations/blob/master/img/fig7.svg)


На текущем этапе можно увидеть, что созданные модели демонстрируют точность, близкую к идеальной. Однако для более глубокого исследования мы обучим модель на протяжении 10 000 эпох, используя следующие гиперпараметры:

- скорость обучения — 0,001;
- размер батча — 1000;
- количество нейронов в среднем слое — 128.


```python
model_2 = train(10_000, 0.001, 1000, Net1, (128,), generateFracData)
```


```python
display.SVG("""<svg viewBox="0 0 899.25 400" xmlns="http://www.w3.org/2000/svg"><g><g><g><g><g><line x1="51.5625" y1="377" x2="46.5625" y2="377" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="345.58333333333337" x2="46.5625" y2="345.58333333333337" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="314.16666666666663" x2="46.5625" y2="314.16666666666663" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="282.75000000000006" x2="46.5625" y2="282.75000000000006" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="251.33333333333337" x2="46.5625" y2="251.33333333333337" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="219.9166666666667" x2="46.5625" y2="219.9166666666667" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="188.50000000000003" x2="46.5625" y2="188.50000000000003" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="157.08333333333337" x2="46.5625" y2="157.08333333333337" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="125.66666666666673" x2="46.5625" y2="125.66666666666673" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="94.25000000000004" x2="46.5625" y2="94.25000000000004" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="62.8333333333334" x2="46.5625" y2="62.8333333333334" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="31.416666666666682" x2="46.5625" y2="31.416666666666682" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="51.5625" y1="0" x2="46.5625" y2="0" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line></g><g transform="translate(41.5625, 0)"><text x="0" y="377" dx="0em" dy="0.3em" style="text-anchor: end; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.45</text><text x="0" y="345.58333333333337" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.5</text><text x="0" y="314.16666666666663" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.55</text><text x="0" y="282.75000000000006" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.6</text><text x="0" y="251.33333333333337" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.65</text><text x="0" y="219.9166666666667" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.7</text><text x="0" y="188.50000000000003" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.75</text><text x="0" y="157.08333333333337" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.8</text><text x="0" y="125.66666666666673" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.85</text><text x="0" y="94.25000000000004" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.9</text><text x="0" y="62.8333333333334" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0.95</text><text x="0" y="31.416666666666682" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">1</text><text x="0" y="0" dx="0em" dy="0.3em" style="text-anchor: end; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">1.05</text></g><line x1="51.5625" y1="0" x2="51.5625" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line></g></g><g transform="translate(51, 0)" clip-path="url(#clip_0)"><clipPath id="clip_0"><rect width="847" height="377"></rect></clipPath><g><g><g><line x1="0" y1="0" x2="0" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="70.640625" y1="0" x2="70.640625" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="141.28125" y1="0" x2="141.28125" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="211.921875" y1="0" x2="211.921875" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="282.5625" y1="0" x2="282.5625" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="353.203125" y1="0" x2="353.203125" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="423.84375" y1="0" x2="423.84375" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="494.48437500000006" y1="0" x2="494.48437500000006" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="565.125" y1="0" x2="565.125" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="635.765625" y1="0" x2="635.765625" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="706.40625" y1="0" x2="706.40625" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="777.046875" y1="0" x2="777.046875" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="847.6875" y1="0" x2="847.6875" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line></g><g><line x1="0" y1="377" x2="847.6875" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="345.58333333333337" x2="847.6875" y2="345.58333333333337" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="314.16666666666663" x2="847.6875" y2="314.16666666666663" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="282.75000000000006" x2="847.6875" y2="282.75000000000006" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="251.33333333333337" x2="847.6875" y2="251.33333333333337" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="219.9166666666667" x2="847.6875" y2="219.9166666666667" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="188.50000000000003" x2="847.6875" y2="188.50000000000003" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="157.08333333333337" x2="847.6875" y2="157.08333333333337" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="125.66666666666673" x2="847.6875" y2="125.66666666666673" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="94.25000000000004" x2="847.6875" y2="94.25000000000004" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="62.8333333333334" x2="847.6875" y2="62.8333333333334" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="31.416666666666682" x2="847.6875" y2="31.416666666666682" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="0" x2="847.6875" y2="0" fill="rgb(0, 0, 0)" stroke="rgb(189, 189, 189)" stroke-width="1px" opacity="0.25"></line></g></g></g><g><g><line x1="0" y1="659.75" x2="847.6875" y2="659.75" fill="rgb(0, 0, 0)" stroke="rgb(153, 153, 153)" stroke-width="1.5px"></line></g></g><g><g><line x1="70.640625" y1="0" x2="70.640625" y2="377" fill="rgb(0, 0, 0)" stroke="rgb(153, 153, 153)" stroke-width="1.5px"></line></g></g><g><g><g><g><g><path stroke="rgb(51, 187, 238)" stroke-width="2px" d="M74.243296875,365.3309549490611L81.23671875000001,318.6547447840373L88.371421875,361.7404714822769L95.36484375,361.7404714822769L102.49954687500001,343.78809159994125L109.49296875000002,329.42619518438977L116.627671875,372.51190315683687L123.62109375000001,315.0642987688383L130.755796875,361.7404714822769L137.74921875,361.7404714822769L144.883921875,185.8071561257045L151.87734375,365.3309549490611L159.01204687499998,42.18807961543404L166.00546875,49.36904654900234L173.140171875,42.18807961543404L180.13359375,289.93095195293427L187.268296875,49.36904654900234L194.26171875,42.18807961543404L201.39642187500002,96.0452567140262L208.38984374999998,307.8833318352699L215.52454687499997,31.416666666666682L222.51796875000002,31.416666666666682L229.65267187499998,35.00715013345086L236.64609375,31.416666666666682L243.78079687500002,354.5595232745012L250.77421875000002,153.49284237623218L257.90892187500003,200.16905254125598L264.90234375,85.27380631367372L272.037046875,38.59763360023503L279.03046875,31.416666666666682L286.165171875,31.416666666666682L293.15859374999997,45.77856308221821L300.293296875,35.00715013345086L307.28671875,42.18807961543404L314.421421875,124.76904954512916L321.41484375,45.77856308221821L328.549546875,42.18807961543404L335.54296875,49.36904654900234L342.677671875,35.00715013345086L349.67109374999995,67.32142643133805L356.805796875,31.416666666666682L363.79921874999997,35.00715013345086L370.933921875,42.18807961543404L377.92734375000003,38.59763360023503L385.06204687499996,31.416666666666682L392.05546875000005,31.416666666666682L399.190171875,31.416666666666682L406.18359375,38.59763360023503L413.318296875,35.00715013345086L420.31171875,35.00715013345086L427.446421875,42.18807961543404L434.43984374999997,45.77856308221821L441.574546875,38.59763360023503L448.56796875000003,35.00715013345086L455.63203124999995,45.77856308221821L462.69609374999993,42.18807961543404L469.83079687500003,35.00715013345086L476.894859375,56.55001348257065L483.958921875,35.00715013345086L490.95234375000007,35.00715013345086L498.087046875,31.416666666666682L505.08046874999997,31.416666666666682L512.215171875,31.416666666666682L519.2085937500001,31.416666666666682L526.343296875,31.416666666666682L533.3367187499999,31.416666666666682L540.4714218749999,35.00715013345086L547.46484375,35.00715013345086L554.599546875,35.00715013345086L561.59296875,31.416666666666682L568.727671875,31.416666666666682L575.72109375,42.18807961543404L582.855796875,35.00715013345086L589.84921875,31.416666666666682L596.9839218750001,31.416666666666682L603.97734375,31.416666666666682L611.112046875,35.00715013345086L618.10546875,35.00715013345086L625.240171875,31.416666666666682L632.2335937500001,31.416666666666682L639.3682968749999,35.00715013345086L646.3617187499999,31.416666666666682L653.496421875,31.416666666666682L660.48984375,35.00715013345086L667.624546875,31.416666666666682L674.6179687499999,35.00715013345086L681.752671875,35.00715013345086L688.74609375,35.00715013345086L695.880796875,45.77856308221821L702.8742187500001,35.00715013345086L710.008921875,31.416666666666682L717.00234375,35.00715013345086L724.137046875,42.18807961543404L731.1304687500001,31.416666666666682L738.265171875,31.416666666666682L745.2585937499999,31.416666666666682L752.3932968749999,42.18807961543404L759.38671875,35.00715013345086L766.521421875,31.416666666666682L773.51484375,31.416666666666682" style="fill: none;" fill="none"></path></g></g></g></g></g></g><g transform="translate(51, 377)" clip-path="url(#clip_1)"><clipPath id="clip_1"><rect width="847" height="23"></rect></clipPath><g><g><line x1="0" y1="0" x2="0" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="70.640625" y1="0" x2="70.640625" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="141.28125" y1="0" x2="141.28125" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="211.921875" y1="0" x2="211.921875" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="282.5625" y1="0" x2="282.5625" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="353.203125" y1="0" x2="353.203125" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="423.84375" y1="0" x2="423.84375" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="494.48437500000006" y1="0" x2="494.48437500000006" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="565.125" y1="0" x2="565.125" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="635.765625" y1="0" x2="635.765625" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="706.40625" y1="0" x2="706.40625" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="777.046875" y1="0" x2="777.046875" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="847.6875" y1="0" x2="847.6875" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line></g><g transform="translate(0, 8)"><text x="0" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">-1k</text><text x="70.640625" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">0</text><text x="141.28125" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">1k</text><text x="211.921875" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">2k</text><text x="282.5625" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">3k</text><text x="353.203125" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">4k</text><text x="423.84375" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">5k</text><text x="494.48437500000006" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">6k</text><text x="565.125" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">7k</text><text x="635.765625" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">8k</text><text x="706.40625" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">9k</text><text x="777.046875" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">10k</text><text x="847.6875" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(255, 255, 255)" stroke="none" stroke-width="1px">11k</text></g><line x1="0" y1="0" x2="847.6875" y2="0" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line></g></g></g></g></svg>""")
```




    
![svg](https://github.com/Cyber-Zhaba/prediction-of-permutations/blob/master/img/fig8.svg)
    



Модель достигает верхнего предела довольно быстро, всего за 4000 эпох. После этого она начинает колебаться в пределах 95%. Давайте оценим точность модели на 1000 выборках из 1000 перестановок.


```python
with torch.no_grad():
    scores = []
    for _ in range(1000):
        x, y = generateFracData(1000, device)
        if y[0] > 0:
            scores.append(int(
                model_2(x).sum() > 0.5
            ))
        else:
            scores.append(int(
                model_2(x).sum() < 0.5
            ))
    print("Final Acc", sum(scores) / len(scores))
```

    Final Acc 0.998
    

Итоговая точность составила 99,8%. Этого более чем достаточно для решения исходной задачи.


```python
with torch.no_grad():
    with open("permutations.in", "r") as file:
        n = int(file.readline())
        randPermList = []
        stupidPermList = []
        for i in range(n):
            x_arr = np.zeros(8)
            for _ in range(1000):
                for j, p in enumerate(list(map(int, file.readline().split()))):
                    if j == p:
                        x_arr[j] += 1 / 1000
            
            x = torch.Tensor(x_arr).to(device)
            pred = model_2(x)

            if pred < 0.5:
                stupidPermList.append(i)
            else:
                randPermList.append(i)
        
        for v in randPermList + stupidPermList:
            print(v, file=open("out.txt", "a"))
```
