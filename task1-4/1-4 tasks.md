### 1. Подготовьте данные для word2vec по одной из недавно прочитанных и обучите модель


```python
import codecs
import numpy as np
import gensim
from gensim.models import Word2Vec

# загрузка файла
with codecs.open('persi.txt', mode='r') as file:
    docs = file.readlines()
    
max_sen_len = 12

# предварительная обработка
sentences = [sent for doc in docs for sent in doc.split('.')]
sentences = [[word for word in sent.lower().split()[:max_sen_len]] for sent in sentences]
print('Всего', len(sentences),'предложений')
```

    Всего 14555 предложений
    


```python
# обучение
model = gensim.models.Word2Vec(sentences, vector_size=100, min_count=1, window=5, epochs=100)

pretrained_weights = model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape
print('Размер словаря:', vocab_size)
print('Размер каждого вектора:', emdedding_size)
```

    Размер словаря: 17208
    Размер каждого вектора: 100
    


```python
sim_words = model.wv.most_similar('человек') 
for words in sim_words:
    print(words)
```

    ('приземистый', 0.7361227869987488)
    ('леопардовой', 0.6976988315582275)
    ('молодой', 0.6477242708206177)
    ('обыкновенный', 0.6444881558418274)
    ('гавайской', 0.6417467594146729)
    ('шум', 0.6335494518280029)
    ('гарлеме', 0.6268101334571838)
    ('черном?', 0.6119715571403503)
    ('школьного', 0.6084041595458984)
    ('отеделение?', 0.5855964422225952)
    


```python
sim_words = model.wv.most_similar('пророчество') 
for words in sim_words:
    print(words)
```

    ('предупреждает', 0.7625859975814819)
    ('ребенке,', 0.6998143792152405)
    ('которому', 0.6756278872489929)
    ('привело', 0.6692213416099548)
    ('долгожданное', 0.660474419593811)
    ('раскрывается', 0.6487653255462646)
    ('видении,которое', 0.6444886326789856)
    ('думал', 0.6348645091056824)
    ('джодж', 0.6334055662155151)
    ('живым,', 0.6330338716506958)
    


```python
sim_words = model.wv.most_similar('герой') 
for words in sim_words:
    print(words)
```

    ('великий', 0.6497102379798889)
    ('пророчества?', 0.607123851776123)
    ('осуждающе', 0.5921017527580261)
    ('орфей', 0.5902287364006042)
    ('оуэн', 0.5865791440010071)
    ('kool-aid,', 0.5812691450119019)
    ('готов,', 0.5771592259407043)
    ('награду,', 0.5759702920913696)
    ('нюхать', 0.5524123311042786)
    ('вечеринка,', 0.5503742694854736)
    


```python
sim_words = model.wv.most_similar('олимп') 
for words in sim_words:
    print(words)
```

    ('грани', 0.6018781661987305)
    ('детонатора,', 0.5897119641304016)
    ('кнопку', 0.5869900584220886)
    ('падения,', 0.5855629444122314)
    ('падет,', 0.5796062350273132)
    ('пешеходов', 0.5691882967948914)
    ('дней,', 0.5627169013023376)
    ('перестроить', 0.5542925596237183)
    ('остался', 0.5461438894271851)
    ('фактически', 0.5447206497192383)
    


```python
sim_words = model.wv.most_similar('кронос') 
for words in sim_words:
    print(words)
```

    ('он', 0.5472914576530457)
    ('здесь!"', 0.5343891978263855)
    ('лука', 0.5259333252906799)
    ('пандоры,', 0.5241441130638123)
    ('громче:', 0.5074095129966736)
    ('аид', 0.5010095238685608)
    ('воздушный', 0.49866342544555664)
    ('тайсон', 0.48851388692855835)
    ('гудзон', 0.4884437620639801)
    ('неправильно,', 0.4840896725654602)
    


```python
vec = model.wv.most_similar(positive=['человек', 'бог'], negative=['бог'])
vec
```




    [('приземистый', 0.7361227869987488),
     ('леопардовой', 0.697698712348938),
     ('молодой', 0.6477242708206177),
     ('обыкновенный', 0.6444880962371826),
     ('гавайской', 0.6417466998100281),
     ('шум', 0.6335494518280029),
     ('гарлеме', 0.6268100738525391),
     ('черном?', 0.6119715571403503),
     ('школьного', 0.6084041595458984),
     ('отеделение?', 0.5855963826179504)]




```python
vec = model.wv.most_similar(positive=['бог', 'олимп'], negative=['аид'])
vec
```




    [('грани', 0.5339633822441101),
     ('падения,', 0.5314422249794006),
     ('различным', 0.5209038853645325),
     ('дней,', 0.5137315988540649),
     ('путешественников', 0.50180584192276),
     ('пережил', 0.48928552865982056),
     ('каждый', 0.47728490829467773),
     ('принцессе', 0.4718187153339386),
     ('нормально,', 0.4694176912307739),
     ('разделиться,', 0.46187666058540344)]



### 2. Для обучения на нефтяных скважин добавьте во входные данные информацию со столбцов Gas, Water (т. е. размер x_data будет (440, 12, 3)) и обучите новую модель. Выход содержит Liquid, Gas (для дальнейшего предсказания). Графики с результатами для Liquid


```python
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv('production.csv')
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>API</th>
      <th>Year</th>
      <th>Month</th>
      <th>Liquid</th>
      <th>Gas</th>
      <th>RatioGasOil</th>
      <th>Water</th>
      <th>PercentWater</th>
      <th>DaysOn</th>
      <th>_LastUpdate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5005072170100</td>
      <td>2014</td>
      <td>11</td>
      <td>9783</td>
      <td>11470</td>
      <td>1.172442</td>
      <td>10558</td>
      <td>1.079219</td>
      <td>14</td>
      <td>2016-04-06 17:20:05.757</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5005072170100</td>
      <td>2014</td>
      <td>12</td>
      <td>24206</td>
      <td>26476</td>
      <td>1.093778</td>
      <td>5719</td>
      <td>0.236264</td>
      <td>31</td>
      <td>2016-04-06 17:20:05.757</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5005072170100</td>
      <td>2015</td>
      <td>1</td>
      <td>20449</td>
      <td>26381</td>
      <td>1.290088</td>
      <td>2196</td>
      <td>0.107389</td>
      <td>31</td>
      <td>2016-04-06 17:20:05.757</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5005072170100</td>
      <td>2015</td>
      <td>2</td>
      <td>6820</td>
      <td>10390</td>
      <td>1.523460</td>
      <td>583</td>
      <td>0.085484</td>
      <td>28</td>
      <td>2016-04-06 17:20:05.757</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5005072170100</td>
      <td>2015</td>
      <td>3</td>
      <td>7349</td>
      <td>7005</td>
      <td>0.953191</td>
      <td>122</td>
      <td>0.016601</td>
      <td>13</td>
      <td>2016-06-16 14:07:33.203</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1195</th>
      <td>49021210550000</td>
      <td>2015</td>
      <td>10</td>
      <td>1262</td>
      <td>665</td>
      <td>0.526941</td>
      <td>341</td>
      <td>0.270206</td>
      <td>31</td>
      <td>2016-04-06 15:40:34.957</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>49021210550000</td>
      <td>2015</td>
      <td>11</td>
      <td>1410</td>
      <td>826</td>
      <td>0.585816</td>
      <td>572</td>
      <td>0.405674</td>
      <td>30</td>
      <td>2016-04-06 15:40:34.957</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>49021210550000</td>
      <td>2015</td>
      <td>12</td>
      <td>1443</td>
      <td>895</td>
      <td>0.620236</td>
      <td>620</td>
      <td>0.429660</td>
      <td>31</td>
      <td>2016-04-06 15:40:34.957</td>
    </tr>
    <tr>
      <th>1198</th>
      <td>49021210550000</td>
      <td>2016</td>
      <td>1</td>
      <td>1654</td>
      <td>988</td>
      <td>0.597340</td>
      <td>808</td>
      <td>0.488513</td>
      <td>31</td>
      <td>2016-04-06 15:40:34.957</td>
    </tr>
    <tr>
      <th>1199</th>
      <td>49021210550000</td>
      <td>2016</td>
      <td>2</td>
      <td>1328</td>
      <td>827</td>
      <td>0.622741</td>
      <td>577</td>
      <td>0.434488</td>
      <td>29</td>
      <td>2016-04-06 15:40:34.957</td>
    </tr>
  </tbody>
</table>
<p>1200 rows × 10 columns</p>
</div>




```python
# Подготовка данных по добыче

liquid = df.groupby('API')['Liquid'].apply(lambda df_: df_.reset_index(drop=True))
gas = df.groupby('API')['Gas'].apply(lambda df_: df_.reset_index(drop=True))
water = df.groupby('API')['Water'].apply(lambda df_: df_.reset_index(drop=True))

```


```python
df_prod = gas.unstack()
df_prod2 = water.unstack()
df_prod3 = liquid.unstack()
```


```python
# Масштабирование
# Gas
data = df_prod.values
data = data / data.max()
data = data[:, :, np.newaxis]
# Water
data2 = df_prod2.values
data2 = data2 / data2.max()
data2 = data2[:, :, np.newaxis]
# Liquid
data3 = df_prod3.values
data3 = data3 / data3.max()
data3 = data3[:, :, np.newaxis]


```


```python
# Деление на трейн/тест (80 на 20)

data_tr_gas = data[:40]
data_tr_water = data2[:40]
data_tr_liquid = data3[:40]

data_tst_gas = data[40:]
data_tst_water = data2[40:]
data_tst_liquid = data3[40:]

# Объединяем
a = np.concatenate([data_tst_gas, data_tst_water], axis=2)
data_tst = np.concatenate([a, data_tst_liquid], axis=2)

```


```python
x_data_liquid = [data_tr_liquid[:, i:i+12] for i in range(11)]
y_data_liquid = [data_tr_liquid[:, i+1:i+13] for i in range(11)]

x_data_water = [data_tr_water[:, i:i+12] for i in range(11)]
y_data_water = [data_tr_water[:, i+1:i+13] for i in range(11)]

x_data_gas = [data_tr_gas[:, i:i+12] for i in range(11)]
y_data_gas = [data_tr_gas[:, i+1:i+13] for i in range(11)]



x_data_liquid = np.concatenate(x_data_liquid, axis=0)
y_data_liquid = np.concatenate(y_data_liquid, axis=0)

x_data_water = np.concatenate(x_data_water, axis=0)
y_data_water = np.concatenate(y_data_water, axis=0)

x_data_gas = np.concatenate(x_data_gas, axis=0)
y_data_gas = np.concatenate(y_data_gas, axis=0)



arr = np.concatenate([x_data_liquid, x_data_water], axis=2)
x_data = np.concatenate([arr, x_data_gas], axis=2)

arr2 = np.concatenate([y_data_liquid, y_data_water], axis=2)
y_data = np.concatenate([arr2, y_data_gas], axis=2)


print(x_data.shape, y_data.shape)
```

    (440, 12, 3) (440, 12, 3)
    


```python
tensor_x = torch.Tensor(x_data) # transform to torch tensor
tensor_y = torch.Tensor(y_data)

oil_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
oil_dataloader = DataLoader(oil_dataset, batch_size=16) # create your dataloader
```


```python
for x_t, y_t in oil_dataloader:
    break
x_t.shape, y_t.shape
```




    (torch.Size([16, 12, 3]), torch.Size([16, 12, 3]))




```python
class OilModel(nn.Module):
    def __init__(self, timesteps=12, units=32):
        super().__init__()
        self.lstm1 = nn.LSTM(3, units, 2, batch_first=True)
        self.dense = nn.Linear(units, 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h, _ = self.lstm1(x)
        outs = []
        for i in range(h.shape[0]):
            outs.append(self.relu(self.dense(h[i])))
        out = torch.stack(outs, dim=0)
        return out
```


```python
model = OilModel()
opt = optim.Adam(model.parameters())
criterion = nn.MSELoss()
```


```python
NUM_EPOCHS = 20

for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    num = 0
    for x_t, y_t in oil_dataloader:
        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = model(x_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        num += 1
        
    print(f'[Epoch: {epoch + 1:2d}] loss: {running_loss / num:.3f}')

print('Finished Training')
```

    [Epoch:  1] loss: 0.013
    [Epoch:  2] loss: 0.011
    [Epoch:  3] loss: 0.011
    [Epoch:  4] loss: 0.010
    [Epoch:  5] loss: 0.009
    [Epoch:  6] loss: 0.009
    [Epoch:  7] loss: 0.009
    [Epoch:  8] loss: 0.009
    [Epoch:  9] loss: 0.008
    [Epoch: 10] loss: 0.008
    [Epoch: 11] loss: 0.008
    [Epoch: 12] loss: 0.008
    [Epoch: 13] loss: 0.008
    [Epoch: 14] loss: 0.008
    [Epoch: 15] loss: 0.008
    [Epoch: 16] loss: 0.008
    [Epoch: 17] loss: 0.008
    [Epoch: 18] loss: 0.008
    [Epoch: 19] loss: 0.008
    [Epoch: 20] loss: 0.008
    Finished Training
    


```python
# Предскажем на год вперёд используя данные только первого года
x_tst = data_tst[:, :12]
predicts = np.zeros((x_tst.shape[0], 0, x_tst.shape[2]))

for i in range(12):
    x = np.concatenate((x_tst[:, i:], predicts), axis=1)
    x_t = torch.from_numpy(x).float()
    pred = model(x_t).detach().numpy()
    last_pred = pred[:, -1:]  # Нас интересует только последний месяц
    predicts = np.concatenate((predicts, last_pred), axis=1)
    
```


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for iapi in range(4):
    plt.subplot(2, 2, iapi+1)
    plt.plot(np.arange(x_tst.shape[1]), x_tst[iapi, :, 0], label='Actual')
    plt.plot(np.arange(predicts.shape[1])+x_tst.shape[1], predicts[iapi, :, 0], label='Prediction')
    plt.legend()
plt.show()
```


    
![png](output_23_0.png)
    


### 3. Из этого же текста (п.1) возьмите небольшой фрагмент, разбейте на предложения с одинаковым числом символов. Каждый символ предложения закодируйте с помощью one hot encoding. В итоге у вас должен получиться массив размера (n_sentences, sentence_len, encoding_size).


```python
import re

with open('persi_.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = re.sub(r'[^А-я ]', '', text) #  оставим в тексте только символы русских букв и символы пробела

text = text.lower() # переводим в нижний регистр

text
```




    'весь год полубоги готовились к сражению против титанов зная что шансы на победу очень малы армия кроноса сильна как никогда и с каждым богом и полукровкой которого он вербует власть злого титана только растет в то время как олимпийцы изо всех сил пытаются сдержать неистовствующего монстра тифона кронос начинает продвижение на ньюйорк где находится абсолютно неохраняемая гора олимп теперь это дело перси джексона и армии молодых полубогов остановить бога времени в этой важной заключительной книге раскрывается долгожданное пророчество о шестнадцатом дне рождения перси и так как сражение за западную цивилизацию бушует на улицах манхэттана перси оказывается перед ужасающим подозрением что он может бороться против своей собственной судьбы'




```python
CHARS  = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "        # алфавит + пробел

#  отображение символов в целые числа
char_to_int = dict((c, i) for i, c in enumerate(CHARS)) 
int_to_char = dict((i, c) for i, c in enumerate(CHARS)) 
# входные данные кодируются в целых числах 
int_encoded = [char_to_int[char] for char in text] 

n = 12 # количество символов в одном предложении

chunks = [int_encoded[i:i+n] for i in range(0, len(int_encoded), n)] 

print('Всего', len(chunks),'предложений')

```

    Всего 62 предложений
    


```python
# one hot encode 
import numpy as np
onehot_encoded = list() 
for value in int_encoded: 
  letter = [0 for _ in range(len(CHARS))] 
  letter[value] = 1 
  onehot_encoded.append(letter) 
onehot_encoded= np.array(onehot_encoded)
onehot_encoded.resize(62, 12, 34)
# по условию массив должен быть размером (n_sentences, sentence_len, encoding_size)
# в данном случае 62 предложения, количество символов в каждом - 12, размер кодировки - 34 (33 буквы алфавита и пробел)
print('Размерность массива:', onehot_encoded.shape)
onehot_encoded

```

    Размерность массива: (62, 12, 34)
    




    array([[[0, 0, 1, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 1, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 1, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [1, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           ...,
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 1, 0],
            [0, 0, 0, ..., 0, 0, 1],
            ...,
            [0, 0, 0, ..., 0, 0, 1],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 1, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 1, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]]])




```python
print(onehot_encoded[0][0])

#например, в тексте первая буква 'в', по порядку в алфовите третья
#но т.к. индексация с нуля, то 'в' = '2' -> поэтому отмечен 2-ой индекс как 1 для 'в'

#вторая буква 'е' - шестая по алфавиту, по индексу - пятая -> поэтому отмечен 5-ой индекс как 1 для ''
#с другими символами аналогично

print(onehot_encoded[0][1])
```

    [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    

### 4. На полученных в п.3 задании обучение модель RNN для предсказания следующего символа. Посмотрите результат при последовательной генерации.


```python

import tensorflow as tf
from tensorflow.keras import layers
recurrent_model = tf.keras.Sequential()
recurrent_model.add(tf.keras.Input((12, 34))) #при тренировке в рекуррентные модели keras подается сразу вся последовательность, поэтому в input теперь два числа. 1-длина последовательности, 2-размер OHE
recurrent_model.add(tf.keras.layers.SimpleRNN(500)) #рекуррентный слой на 500 нейронов
recurrent_model.add(tf.keras.layers.Dense(34, activation='softmax'))
recurrent_model.summary()

onehot_encoded.resize(744, 34)
n = onehot_encoded.shape[0]-12
X = np.array([onehot_encoded[i:i+12, :] for i in range(n)])
Y = onehot_encoded[12:] 

recurrent_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = recurrent_model.fit(X, Y, batch_size=32, epochs=200)


```

    Model: "sequential_8"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     simple_rnn_8 (SimpleRNN)    (None, 500)               267500    
                                                                     
     dense_8 (Dense)             (None, 34)                17034     
                                                                     
    =================================================================
    Total params: 284,534
    Trainable params: 284,534
    Non-trainable params: 0
    _________________________________________________________________
    (732, 12, 34)
    (732, 34)
    Epoch 1/200
    23/23 [==============================] - 3s 36ms/step - loss: 3.2676 - accuracy: 0.1038
    Epoch 2/200
    23/23 [==============================] - 1s 31ms/step - loss: 2.6557 - accuracy: 0.2514
    Epoch 3/200
    23/23 [==============================] - 1s 29ms/step - loss: 2.3420 - accuracy: 0.3060
    Epoch 4/200
    23/23 [==============================] - 1s 30ms/step - loss: 2.0846 - accuracy: 0.3880
    Epoch 5/200
    23/23 [==============================] - 1s 30ms/step - loss: 1.9219 - accuracy: 0.4399
    Epoch 6/200
    23/23 [==============================] - 1s 30ms/step - loss: 1.7694 - accuracy: 0.4795
    Epoch 7/200
    23/23 [==============================] - 1s 31ms/step - loss: 1.6428 - accuracy: 0.5205
    Epoch 8/200
    23/23 [==============================] - 1s 29ms/step - loss: 1.5263 - accuracy: 0.5478
    Epoch 9/200
    23/23 [==============================] - 1s 29ms/step - loss: 1.3863 - accuracy: 0.5779
    Epoch 10/200
    23/23 [==============================] - 1s 29ms/step - loss: 1.2911 - accuracy: 0.6148
    Epoch 11/200
    23/23 [==============================] - 1s 31ms/step - loss: 1.0912 - accuracy: 0.6721
    Epoch 12/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.9319 - accuracy: 0.7486
    Epoch 13/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.7891 - accuracy: 0.7828
    Epoch 14/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.6102 - accuracy: 0.8579
    Epoch 15/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.4938 - accuracy: 0.9003
    Epoch 16/200
    23/23 [==============================] - 1s 34ms/step - loss: 0.4438 - accuracy: 0.8989
    Epoch 17/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.3605 - accuracy: 0.9344
    Epoch 18/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.2511 - accuracy: 0.9563
    Epoch 19/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.2098 - accuracy: 0.9549
    Epoch 20/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.1829 - accuracy: 0.9727
    Epoch 21/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.1957 - accuracy: 0.9522
    Epoch 22/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.1362 - accuracy: 0.9795
    Epoch 23/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.1082 - accuracy: 0.9822
    Epoch 24/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0598 - accuracy: 0.9891
    Epoch 25/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0564 - accuracy: 0.9904
    Epoch 26/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.0360 - accuracy: 0.9904
    Epoch 27/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.0645 - accuracy: 0.9809
    Epoch 28/200
    23/23 [==============================] - 1s 32ms/step - loss: 0.0586 - accuracy: 0.9836
    Epoch 29/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.0825 - accuracy: 0.9863
    Epoch 30/200
    23/23 [==============================] - 1s 32ms/step - loss: 0.0529 - accuracy: 0.9877
    Epoch 31/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0543 - accuracy: 0.9877
    Epoch 32/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0722 - accuracy: 0.9891
    Epoch 33/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.0978 - accuracy: 0.9836
    Epoch 34/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0819 - accuracy: 0.9809
    Epoch 35/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0848 - accuracy: 0.9836
    Epoch 36/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0634 - accuracy: 0.9877
    Epoch 37/200
    23/23 [==============================] - 1s 32ms/step - loss: 0.1022 - accuracy: 0.9768
    Epoch 38/200
    23/23 [==============================] - 1s 32ms/step - loss: 0.1299 - accuracy: 0.9768
    Epoch 39/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0633 - accuracy: 0.9891
    Epoch 40/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0465 - accuracy: 0.9891
    Epoch 41/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0396 - accuracy: 0.9918
    Epoch 42/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0291 - accuracy: 0.9945
    Epoch 43/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0825 - accuracy: 0.9891
    Epoch 44/200
    23/23 [==============================] - 1s 35ms/step - loss: 0.0553 - accuracy: 0.9918
    Epoch 45/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.1045 - accuracy: 0.9754
    Epoch 46/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.2109 - accuracy: 0.9372
    Epoch 47/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.5783 - accuracy: 0.8238
    Epoch 48/200
    23/23 [==============================] - 1s 31ms/step - loss: 1.3294 - accuracy: 0.5956
    Epoch 49/200
    23/23 [==============================] - 1s 30ms/step - loss: 1.6677 - accuracy: 0.5041
    Epoch 50/200
    23/23 [==============================] - 1s 30ms/step - loss: 1.6230 - accuracy: 0.5164
    Epoch 51/200
    23/23 [==============================] - 1s 31ms/step - loss: 1.2930 - accuracy: 0.6052
    Epoch 52/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.9041 - accuracy: 0.7022
    Epoch 53/200
    23/23 [==============================] - 1s 32ms/step - loss: 0.5188 - accuracy: 0.8415
    Epoch 54/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.2571 - accuracy: 0.9344
    Epoch 55/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.1151 - accuracy: 0.9727
    Epoch 56/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0467 - accuracy: 0.9891
    Epoch 57/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0313 - accuracy: 0.9904
    Epoch 58/200
    23/23 [==============================] - 1s 32ms/step - loss: 0.0398 - accuracy: 0.9891
    Epoch 59/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.0239 - accuracy: 0.9959
    Epoch 60/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0182 - accuracy: 0.9932
    Epoch 61/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0117 - accuracy: 0.9945
    Epoch 62/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0044 - accuracy: 0.9986
    Epoch 63/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0037 - accuracy: 0.9973
    Epoch 64/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0033 - accuracy: 0.9986
    Epoch 65/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0031 - accuracy: 0.9986
    Epoch 66/200
    23/23 [==============================] - 1s 33ms/step - loss: 0.0029 - accuracy: 0.9986
    Epoch 67/200
    23/23 [==============================] - 1s 32ms/step - loss: 0.0028 - accuracy: 0.9986
    Epoch 68/200
    23/23 [==============================] - 1s 34ms/step - loss: 0.0026 - accuracy: 0.9986
    Epoch 69/200
    23/23 [==============================] - 1s 32ms/step - loss: 0.0025 - accuracy: 0.9986
    Epoch 70/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0024 - accuracy: 0.9973
    Epoch 71/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0023 - accuracy: 0.9973
    Epoch 72/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0022 - accuracy: 0.9973
    Epoch 73/200
    23/23 [==============================] - 1s 32ms/step - loss: 0.0021 - accuracy: 0.9973
    Epoch 74/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0021 - accuracy: 0.9973
    Epoch 75/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0020 - accuracy: 0.9973
    Epoch 76/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0019 - accuracy: 0.9973
    Epoch 77/200
    23/23 [==============================] - 1s 28ms/step - loss: 0.0019 - accuracy: 0.9973
    Epoch 78/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0018 - accuracy: 0.9973
    Epoch 79/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0018 - accuracy: 0.9973
    Epoch 80/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0017 - accuracy: 0.9973
    Epoch 81/200
    23/23 [==============================] - 1s 27ms/step - loss: 0.0017 - accuracy: 0.9973
    Epoch 82/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0016 - accuracy: 0.9973
    Epoch 83/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0016 - accuracy: 0.9973
    Epoch 84/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.0016 - accuracy: 0.9973
    Epoch 85/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0015 - accuracy: 0.9973
    Epoch 86/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0015 - accuracy: 0.9973
    Epoch 87/200
    23/23 [==============================] - 1s 27ms/step - loss: 0.0015 - accuracy: 0.9973
    Epoch 88/200
    23/23 [==============================] - 1s 32ms/step - loss: 0.0014 - accuracy: 0.9973
    Epoch 89/200
    23/23 [==============================] - 1s 30ms/step - loss: 0.0014 - accuracy: 0.9973
    Epoch 90/200
    23/23 [==============================] - 1s 27ms/step - loss: 0.0014 - accuracy: 0.9973
    Epoch 91/200
    23/23 [==============================] - 1s 27ms/step - loss: 0.0013 - accuracy: 0.9973
    Epoch 92/200
    23/23 [==============================] - 1s 28ms/step - loss: 0.0013 - accuracy: 0.9973
    Epoch 93/200
    23/23 [==============================] - 1s 28ms/step - loss: 0.0013 - accuracy: 0.9973
    Epoch 94/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0013 - accuracy: 0.9973
    Epoch 95/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0013 - accuracy: 0.9973
    Epoch 96/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0012 - accuracy: 0.9973
    Epoch 97/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0012 - accuracy: 0.9973
    Epoch 98/200
    23/23 [==============================] - 1s 28ms/step - loss: 0.0012 - accuracy: 0.9973
    Epoch 99/200
    23/23 [==============================] - 1s 28ms/step - loss: 0.0011 - accuracy: 0.9973
    Epoch 100/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0011 - accuracy: 0.9986
    Epoch 101/200
    23/23 [==============================] - 1s 31ms/step - loss: 0.0011 - accuracy: 0.9986
    Epoch 102/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0011 - accuracy: 0.9986
    Epoch 103/200
    23/23 [==============================] - 1s 28ms/step - loss: 0.0011 - accuracy: 0.9986
    Epoch 104/200
    23/23 [==============================] - 1s 33ms/step - loss: 0.0011 - accuracy: 0.9986
    Epoch 105/200
    23/23 [==============================] - 1s 33ms/step - loss: 0.0010 - accuracy: 0.9986
    Epoch 106/200
    23/23 [==============================] - 1s 29ms/step - loss: 0.0010 - accuracy: 0.9986
    Epoch 107/200
    23/23 [==============================] - 1s 28ms/step - loss: 9.9686e-04 - accuracy: 0.9986
    Epoch 108/200
    23/23 [==============================] - 1s 31ms/step - loss: 9.9072e-04 - accuracy: 0.9986
    Epoch 109/200
    23/23 [==============================] - 1s 29ms/step - loss: 9.6518e-04 - accuracy: 0.9986
    Epoch 110/200
    23/23 [==============================] - 1s 29ms/step - loss: 9.4556e-04 - accuracy: 0.9986
    Epoch 111/200
    23/23 [==============================] - 1s 29ms/step - loss: 9.5081e-04 - accuracy: 0.9973
    Epoch 112/200
    23/23 [==============================] - 1s 30ms/step - loss: 9.5223e-04 - accuracy: 0.9973
    Epoch 113/200
    23/23 [==============================] - 1s 29ms/step - loss: 9.1225e-04 - accuracy: 0.9973
    Epoch 114/200
    23/23 [==============================] - 1s 28ms/step - loss: 8.8342e-04 - accuracy: 0.9973
    Epoch 115/200
    23/23 [==============================] - 1s 30ms/step - loss: 8.8866e-04 - accuracy: 0.9973
    Epoch 116/200
    23/23 [==============================] - 1s 28ms/step - loss: 8.8184e-04 - accuracy: 0.9973
    Epoch 117/200
    23/23 [==============================] - 1s 27ms/step - loss: 8.6799e-04 - accuracy: 0.9973
    Epoch 118/200
    23/23 [==============================] - 1s 28ms/step - loss: 8.4648e-04 - accuracy: 0.9973
    Epoch 119/200
    23/23 [==============================] - 1s 28ms/step - loss: 8.2439e-04 - accuracy: 0.9973
    Epoch 120/200
    23/23 [==============================] - 1s 28ms/step - loss: 8.2186e-04 - accuracy: 0.9973
    Epoch 121/200
    23/23 [==============================] - 1s 28ms/step - loss: 8.0483e-04 - accuracy: 0.9973
    Epoch 122/200
    23/23 [==============================] - 1s 28ms/step - loss: 7.8904e-04 - accuracy: 0.9973
    Epoch 123/200
    23/23 [==============================] - 1s 29ms/step - loss: 7.8573e-04 - accuracy: 0.9973
    Epoch 124/200
    23/23 [==============================] - 1s 28ms/step - loss: 7.7250e-04 - accuracy: 0.9973
    Epoch 125/200
    23/23 [==============================] - 1s 28ms/step - loss: 7.5503e-04 - accuracy: 0.9973
    Epoch 126/200
    23/23 [==============================] - 1s 28ms/step - loss: 7.4064e-04 - accuracy: 0.9973
    Epoch 127/200
    23/23 [==============================] - 1s 28ms/step - loss: 7.3795e-04 - accuracy: 0.9973
    Epoch 128/200
    23/23 [==============================] - 1s 31ms/step - loss: 7.3535e-04 - accuracy: 0.9973
    Epoch 129/200
    23/23 [==============================] - 1s 28ms/step - loss: 7.1807e-04 - accuracy: 0.9973
    Epoch 130/200
    23/23 [==============================] - 1s 28ms/step - loss: 7.0344e-04 - accuracy: 0.9973
    Epoch 131/200
    23/23 [==============================] - 1s 28ms/step - loss: 6.9687e-04 - accuracy: 0.9973
    Epoch 132/200
    23/23 [==============================] - 1s 27ms/step - loss: 6.8234e-04 - accuracy: 0.9973
    Epoch 133/200
    23/23 [==============================] - 1s 29ms/step - loss: 6.7176e-04 - accuracy: 0.9973
    Epoch 134/200
    23/23 [==============================] - 1s 27ms/step - loss: 6.6927e-04 - accuracy: 0.9973
    Epoch 135/200
    23/23 [==============================] - 1s 27ms/step - loss: 6.5791e-04 - accuracy: 0.9973
    Epoch 136/200
    23/23 [==============================] - 1s 30ms/step - loss: 6.4675e-04 - accuracy: 0.9973
    Epoch 137/200
    23/23 [==============================] - 1s 28ms/step - loss: 6.2789e-04 - accuracy: 0.9973
    Epoch 138/200
    23/23 [==============================] - 1s 30ms/step - loss: 6.2406e-04 - accuracy: 0.9973
    Epoch 139/200
    23/23 [==============================] - 1s 32ms/step - loss: 6.1749e-04 - accuracy: 0.9973
    Epoch 140/200
    23/23 [==============================] - 1s 33ms/step - loss: 6.1398e-04 - accuracy: 0.9973
    Epoch 141/200
    23/23 [==============================] - 1s 34ms/step - loss: 6.0057e-04 - accuracy: 0.9973
    Epoch 142/200
    23/23 [==============================] - 1s 31ms/step - loss: 5.9529e-04 - accuracy: 0.9973
    Epoch 143/200
    23/23 [==============================] - 1s 32ms/step - loss: 5.9281e-04 - accuracy: 0.9973
    Epoch 144/200
    23/23 [==============================] - 1s 29ms/step - loss: 5.7642e-04 - accuracy: 0.9973
    Epoch 145/200
    23/23 [==============================] - 1s 31ms/step - loss: 5.5985e-04 - accuracy: 0.9973
    Epoch 146/200
    23/23 [==============================] - 1s 30ms/step - loss: 5.6048e-04 - accuracy: 0.9973
    Epoch 147/200
    23/23 [==============================] - 1s 30ms/step - loss: 5.5273e-04 - accuracy: 0.9973
    Epoch 148/200
    23/23 [==============================] - 1s 29ms/step - loss: 5.3541e-04 - accuracy: 0.9973
    Epoch 149/200
    23/23 [==============================] - 1s 29ms/step - loss: 5.2819e-04 - accuracy: 0.9973
    Epoch 150/200
    23/23 [==============================] - 1s 30ms/step - loss: 5.2673e-04 - accuracy: 0.9973
    Epoch 151/200
    23/23 [==============================] - 1s 29ms/step - loss: 5.2135e-04 - accuracy: 0.9973
    Epoch 152/200
    23/23 [==============================] - 1s 30ms/step - loss: 5.1254e-04 - accuracy: 0.9973
    Epoch 153/200
    23/23 [==============================] - 1s 29ms/step - loss: 5.0274e-04 - accuracy: 0.9973
    Epoch 154/200
    23/23 [==============================] - 1s 29ms/step - loss: 4.9403e-04 - accuracy: 0.9973
    Epoch 155/200
    23/23 [==============================] - 1s 28ms/step - loss: 4.8777e-04 - accuracy: 0.9973
    Epoch 156/200
    23/23 [==============================] - 1s 29ms/step - loss: 4.8306e-04 - accuracy: 0.9973
    Epoch 157/200
    23/23 [==============================] - 1s 29ms/step - loss: 4.7886e-04 - accuracy: 0.9973
    Epoch 158/200
    23/23 [==============================] - 1s 34ms/step - loss: 4.6435e-04 - accuracy: 0.9973
    Epoch 159/200
    23/23 [==============================] - 1s 31ms/step - loss: 4.5709e-04 - accuracy: 0.9973
    Epoch 160/200
    23/23 [==============================] - 1s 29ms/step - loss: 4.5154e-04 - accuracy: 0.9973
    Epoch 161/200
    23/23 [==============================] - 1s 28ms/step - loss: 4.5329e-04 - accuracy: 0.9973
    Epoch 162/200
    23/23 [==============================] - 1s 31ms/step - loss: 4.3833e-04 - accuracy: 0.9973
    Epoch 163/200
    23/23 [==============================] - 1s 29ms/step - loss: 4.3129e-04 - accuracy: 0.9973
    Epoch 164/200
    23/23 [==============================] - 1s 31ms/step - loss: 4.2642e-04 - accuracy: 0.9973
    Epoch 165/200
    23/23 [==============================] - 1s 29ms/step - loss: 4.2448e-04 - accuracy: 0.9973
    Epoch 166/200
    23/23 [==============================] - 1s 29ms/step - loss: 4.1733e-04 - accuracy: 0.9973
    Epoch 167/200
    23/23 [==============================] - 1s 29ms/step - loss: 4.1167e-04 - accuracy: 0.9973
    Epoch 168/200
    23/23 [==============================] - 1s 28ms/step - loss: 4.0128e-04 - accuracy: 0.9973
    Epoch 169/200
    23/23 [==============================] - 1s 29ms/step - loss: 4.0189e-04 - accuracy: 0.9973
    Epoch 170/200
    23/23 [==============================] - 1s 29ms/step - loss: 3.9276e-04 - accuracy: 0.9973
    Epoch 171/200
    23/23 [==============================] - 1s 30ms/step - loss: 3.8726e-04 - accuracy: 0.9973
    Epoch 172/200
    23/23 [==============================] - 1s 30ms/step - loss: 3.8229e-04 - accuracy: 0.9973
    Epoch 173/200
    23/23 [==============================] - 1s 30ms/step - loss: 3.7150e-04 - accuracy: 0.9973
    Epoch 174/200
    23/23 [==============================] - 1s 30ms/step - loss: 3.6634e-04 - accuracy: 0.9973
    Epoch 175/200
    23/23 [==============================] - 1s 30ms/step - loss: 3.6288e-04 - accuracy: 0.9973
    Epoch 176/200
    23/23 [==============================] - 1s 29ms/step - loss: 3.5906e-04 - accuracy: 0.9973
    Epoch 177/200
    23/23 [==============================] - 1s 29ms/step - loss: 3.5637e-04 - accuracy: 0.9973
    Epoch 178/200
    23/23 [==============================] - 1s 31ms/step - loss: 3.4797e-04 - accuracy: 0.9973
    Epoch 179/200
    23/23 [==============================] - 1s 32ms/step - loss: 3.4056e-04 - accuracy: 0.9973
    Epoch 180/200
    23/23 [==============================] - 1s 36ms/step - loss: 3.3753e-04 - accuracy: 0.9973
    Epoch 181/200
    23/23 [==============================] - 1s 29ms/step - loss: 3.2948e-04 - accuracy: 0.9973
    Epoch 182/200
    23/23 [==============================] - 1s 29ms/step - loss: 3.2411e-04 - accuracy: 0.9973
    Epoch 183/200
    23/23 [==============================] - 1s 30ms/step - loss: 3.2700e-04 - accuracy: 0.9973
    Epoch 184/200
    23/23 [==============================] - 1s 33ms/step - loss: 3.2372e-04 - accuracy: 0.9973
    Epoch 185/200
    23/23 [==============================] - 1s 31ms/step - loss: 3.1198e-04 - accuracy: 0.9973
    Epoch 186/200
    23/23 [==============================] - 1s 29ms/step - loss: 3.0391e-04 - accuracy: 0.9973
    Epoch 187/200
    23/23 [==============================] - 1s 32ms/step - loss: 3.0283e-04 - accuracy: 0.9973
    Epoch 188/200
    23/23 [==============================] - 1s 37ms/step - loss: 3.0186e-04 - accuracy: 0.9973
    Epoch 189/200
    23/23 [==============================] - 1s 34ms/step - loss: 2.9555e-04 - accuracy: 0.9973
    Epoch 190/200
    23/23 [==============================] - 1s 34ms/step - loss: 2.8808e-04 - accuracy: 0.9973
    Epoch 191/200
    23/23 [==============================] - 1s 34ms/step - loss: 2.9027e-04 - accuracy: 0.9973
    Epoch 192/200
    23/23 [==============================] - 1s 34ms/step - loss: 2.9432e-04 - accuracy: 0.9973
    Epoch 193/200
    23/23 [==============================] - 1s 29ms/step - loss: 2.7370e-04 - accuracy: 0.9973
    Epoch 194/200
    23/23 [==============================] - 1s 30ms/step - loss: 2.6809e-04 - accuracy: 0.9973
    Epoch 195/200
    23/23 [==============================] - 1s 39ms/step - loss: 2.7117e-04 - accuracy: 0.9973
    Epoch 196/200
    23/23 [==============================] - 1s 33ms/step - loss: 2.8784e-04 - accuracy: 0.9973
    Epoch 197/200
    23/23 [==============================] - 1s 30ms/step - loss: 2.6025e-04 - accuracy: 0.9973
    Epoch 198/200
    23/23 [==============================] - 1s 28ms/step - loss: 2.6004e-04 - accuracy: 0.9973
    Epoch 199/200
    23/23 [==============================] - 1s 28ms/step - loss: 2.7071e-04 - accuracy: 0.9973
    Epoch 200/200
    23/23 [==============================] - 1s 29ms/step - loss: 2.5161e-04 - accuracy: 0.9973
    


```python

```


```python
'''def buildPhrase(inp_str, str_len = 50):
  for i in range(str_len):
    x = []
    for j in range(i, i+12):
      x.append(tokenizer.texts_to_matrix(inp_str[j])) # преобразуем символы в One-Hot-encoding
 
    x = np.array(x)
    inp = x.reshape(1, inp_chars, num_characters)
 
    pred = model.predict( inp ) 
    d = tokenizer.index_word[pred.argmax(axis=1)[0]] # получаем ответ в символьном представлении
 
    inp_str += d # дописываем строку
 
  return inp_str
  
  '''
```




    <bound method Model.predict of <keras.engine.sequential.Sequential object at 0x000001EEB00E7A30>>




```python

```
