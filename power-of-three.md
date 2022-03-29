{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.7.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"markdown","source":"## КРАТКОЕ ОПИСАНИЕ И ЦЕЛИ\n\nКитов и дельфинов в этом наборе данных можно идентифицировать по формам, особенностям и отметинам (некоторые естественные, некоторые приобретенные) спинных плавников, спин, голов и боков. Некоторые виды и некоторые особи имеют очень отчетливые черты, другие гораздо менее отчетливы. Кроме того, индивидуальные особенности могут меняться с течением времени. В этом  датасете содержатся изображения более 15 000 уникальных особей морских млекопитающих из 30 различных видов, собранные из 28 различных исследовательских организаций. \n\nОтдельные особи были идентифицированы вручную и им присвоен индивидуальный идентификатор компанией marine researches, **задача** - правильно идентифицировать этих особей на изображениях. Это сложная задача, которая потенциально может привести к значительному прогрессу в понимании и защите морских млекопитающих по всему миру.\n\n\n~ Ariadna, Sabina and Sonya","metadata":{"execution":{"iopub.status.busy":"2022-03-29T12:44:50.363155Z","iopub.execute_input":"2022-03-29T12:44:50.363505Z","iopub.status.idle":"2022-03-29T12:44:50.374429Z","shell.execute_reply.started":"2022-03-29T12:44:50.363473Z","shell.execute_reply":"2022-03-29T12:44:50.372751Z"}}},{"cell_type":"code","source":"# библиотеки\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport cv2 # work with images\nimport matplotlib.pyplot as plt #visualization\nimport seaborn as sns #visualization\nimport os #path finder\n\nfrom skimage import io\nfrom skimage.color import rgb2gray\nimport plotly.express as px","metadata":{"execution":{"iopub.status.busy":"2022-03-29T14:39:01.372327Z","iopub.execute_input":"2022-03-29T14:39:01.372861Z","iopub.status.idle":"2022-03-29T14:39:01.378794Z","shell.execute_reply.started":"2022-03-29T14:39:01.372827Z","shell.execute_reply":"2022-03-29T14:39:01.377635Z"},"trusted":true},"execution_count":44,"outputs":[]},{"cell_type":"markdown","source":"","metadata":{}},{"cell_type":"code","source":"# получение данных\npath = '/kaggle/input/happy-whale-and-dolphin/'\n\nsample_submission_path = \"/kaggle/input/happy-whale-and-dolphin/sample_submission.csv\"\ntrain_images_path = \"/kaggle/input/happy-whale-and-dolphin/train_images/\"\ntrain_path = \"/kaggle/input/happy-whale-and-dolphin/train.csv\"\ntest_images_path = \"/kaggle/input/happy-whale-and-dolphin/test_images/\"\n\nos.listdir(path)","metadata":{"_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5","_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19","execution":{"iopub.status.busy":"2022-03-29T14:39:01.440841Z","iopub.execute_input":"2022-03-29T14:39:01.441415Z","iopub.status.idle":"2022-03-29T14:39:01.460489Z","shell.execute_reply.started":"2022-03-29T14:39:01.441380Z","shell.execute_reply":"2022-03-29T14:39:01.459083Z"},"trusted":true},"execution_count":45,"outputs":[]},{"cell_type":"markdown","source":"# Работа с данными","metadata":{}},{"cell_type":"code","source":"train_data = pd.read_csv(train_path)\ntrain_data.head()","metadata":{"execution":{"iopub.status.busy":"2022-03-29T14:39:01.490580Z","iopub.execute_input":"2022-03-29T14:39:01.491445Z","iopub.status.idle":"2022-03-29T14:39:01.583286Z","shell.execute_reply.started":"2022-03-29T14:39:01.491406Z","shell.execute_reply":"2022-03-29T14:39:01.581245Z"},"trusted":true},"execution_count":46,"outputs":[]},{"cell_type":"markdown","source":"### Пример того, что должно получится в итоге","metadata":{}},{"cell_type":"code","source":"sample_submission = pd.read_csv(sample_submission_path)\nsample_submission.head()\n","metadata":{"execution":{"iopub.status.busy":"2022-03-29T14:39:01.585367Z","iopub.execute_input":"2022-03-29T14:39:01.585858Z","iopub.status.idle":"2022-03-29T14:39:01.637507Z","shell.execute_reply.started":"2022-03-29T14:39:01.585812Z","shell.execute_reply":"2022-03-29T14:39:01.636279Z"},"trusted":true},"execution_count":47,"outputs":[]},{"cell_type":"code","source":"sample_submission.predictions[0]\n\n#получается для каждого изображения нам нужно пять предсказаний","metadata":{"execution":{"iopub.status.busy":"2022-03-29T14:39:01.638815Z","iopub.execute_input":"2022-03-29T14:39:01.639225Z","iopub.status.idle":"2022-03-29T14:39:01.645715Z","shell.execute_reply.started":"2022-03-29T14:39:01.639193Z","shell.execute_reply":"2022-03-29T14:39:01.644695Z"},"trusted":true},"execution_count":48,"outputs":[]},{"cell_type":"code","source":"train_data.info()","metadata":{"execution":{"iopub.status.busy":"2022-03-29T14:39:01.647817Z","iopub.execute_input":"2022-03-29T14:39:01.648249Z","iopub.status.idle":"2022-03-29T14:39:01.688855Z","shell.execute_reply.started":"2022-03-29T14:39:01.648212Z","shell.execute_reply":"2022-03-29T14:39:01.687676Z"},"trusted":true},"execution_count":49,"outputs":[]},{"cell_type":"markdown","source":"### в тренировачном датасете 51033 входных данных","metadata":{}},{"cell_type":"code","source":"# посмотрим сколько уникальных видов\n\nprint(f'Всего : {train_data[\"species\"].nunique()}')\n","metadata":{"execution":{"iopub.status.busy":"2022-03-29T14:39:24.454056Z","iopub.execute_input":"2022-03-29T14:39:24.454918Z","iopub.status.idle":"2022-03-29T14:39:24.467193Z","shell.execute_reply.started":"2022-03-29T14:39:24.454865Z","shell.execute_reply":"2022-03-29T14:39:24.465757Z"},"trusted":true},"execution_count":51,"outputs":[]},{"cell_type":"markdown","source":"### Наблюдения\n\n1. при рассмотрении данных мы заметили пару орфографических ошибок, поэтому совершим объединение. \n2. также оказалось,что pilot_whale и globis оба являются short_finned_pilot_whale, поэтому они могут быть объединены \n\n*(информация взята из обссуждений, создатель соревнования подтвердил этот факт https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/305468#1677103)","metadata":{}},{"cell_type":"code","source":"train_data['species'].replace({'bottlenose_dolpin':'bottlenose_dolphin',  #missing the 'h' in dolphin\n                               'kiler_whale': 'killer_whale',             #missing the 'l' in killer\n                               'globis':'short_finned_pilot_whale',       #correcting species names\n                               'pilot_whale':'short_finned_pilot_whale'}, #correcting species names\n                              inplace=True)\n\nprint(f'Тогда всего : {train_data[\"species\"].nunique()}')","metadata":{"execution":{"iopub.status.busy":"2022-03-29T15:08:30.434274Z","iopub.execute_input":"2022-03-29T15:08:30.434586Z","iopub.status.idle":"2022-03-29T15:08:30.464648Z","shell.execute_reply.started":"2022-03-29T15:08:30.434551Z","shell.execute_reply":"2022-03-29T15:08:30.463632Z"},"trusted":true},"execution_count":61,"outputs":[]},{"cell_type":"code","source":"print('Количество особей каждого вида')\nprint(train_data['species'].value_counts())\n\n\nprint(\"Количество уникальных идентификаторов\")\nprint(train_data['individual_id'].value_counts())","metadata":{"execution":{"iopub.status.busy":"2022-03-29T15:08:40.742019Z","iopub.execute_input":"2022-03-29T15:08:40.743040Z","iopub.status.idle":"2022-03-29T15:08:40.780042Z","shell.execute_reply.started":"2022-03-29T15:08:40.742893Z","shell.execute_reply":"2022-03-29T15:08:40.778743Z"},"trusted":true},"execution_count":62,"outputs":[]},{"cell_type":"markdown","source":"## Вывод\n\n1. У некоторых видов гораздо больше изображений чем у других.\n2. некоторые id были записаны много раз (до 400 изображений) в то время как другие были зарегистрированы только один раз","metadata":{}},{"cell_type":"markdown","source":"## Визуализируем число особей для каждого вида","metadata":{}},{"cell_type":"code","source":"fig = plt.figure(figsize=(15, 5))\nsns.countplot(x=train_data['species'],\n            order=train_data['species'].value_counts().index).set(title='Species Counts')\nplt.xticks(rotation=90);","metadata":{"execution":{"iopub.status.busy":"2022-03-29T15:08:46.137388Z","iopub.execute_input":"2022-03-29T15:08:46.138070Z","iopub.status.idle":"2022-03-29T15:08:46.588194Z","shell.execute_reply.started":"2022-03-29T15:08:46.138022Z","shell.execute_reply":"2022-03-29T15:08:46.587416Z"},"trusted":true},"execution_count":63,"outputs":[]},{"cell_type":"markdown","source":"### Пока все...\n\n### В планах создать новый столбец, в котором будет хранится информация о том, к какому семейству относится вид (дельфин или кит). Для этого необходимо провести небольшое исследования, чтобы правильно разделить виды на эти две группы","metadata":{}},{"cell_type":"markdown","source":"","metadata":{}}]}