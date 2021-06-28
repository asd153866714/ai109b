# IMDB 影評情緒分析

## 簡介

使用 IMDB 影評資料集做情緒分析，並根據結果判斷準確率

[參考程式碼來源](https://www.kaggle.com/avnika22/imdb-perform-sentiment-analysis-with-scikit-learn)

## 資料集

- [IMDB dataset (Sentiment analysis) in CSV format](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format)

## 使用套件

- numpy
- pandas
- matplotlib --- 繪圖、呈現數據
- scikit‑learn --- 機器學習工具包
- nltk --- 自然語言處理工具

## TF-IDF 演算法

TF-IDF 可以分成兩個部分：**TF 詞頻**、**IDF 逆向文件頻續**

### 計算 TF(term frequency)：詞頻

計算「詞」出現在該文件中的次數，第 t 個詞在第 d 個文件出現的次數用 nt,d 表示。

![image](https://miro.medium.com/max/700/1*OPZc5KxhwGtZYOf2hPWj7w.png)

[來源](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E6%87%89%E7%94%A8-%E5%9E%83%E5%9C%BE%E8%A8%8A%E6%81%AF%E5%81%B5%E6%B8%AC-%E8%88%87-tf-idf%E4%BB%8B%E7%B4%B9-%E5%90%AB%E7%AF%84%E4%BE%8B%E7%A8%8B%E5%BC%8F-2cddc7f7b2c5)


### 計算 IDF(inverse document frequency)：逆向文件頻率

假設  D  是「所有的文件總數」，「詞 t」代表在總共在 dt 篇文章中出現過，則「詞 t」的 IDF 定義為：

![image](https://miro.medium.com/max/166/1*1HNw7mmXnA_BqRt0r2LvPg.png)

由公式可以得知「詞」在越多「文件」中出現代表，相對應的 idf 會比較小

### 計算 TF-IDF：

TF-IDF 就是透過 TF 和 IDF 算每一個「詞」對每一篇「文件」的分數(score)，定義為：

![image](https://miro.medium.com/max/237/1*ftlOXgoIe3W6LL3fmFEhOg.png)

假設「詞 t」很常出現在「文件 d」，那 tft,d 會大，相對的「詞 t」很少出現在其他「文件」中，idft 就也會大，所以整體 scoret,d 也會大

![image](https://miro.medium.com/max/700/1*Z4jV6Bfu4T1eEj5f7UkKMQ.png)

## 程式碼

引入套件

```py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
```

讀取資料集

```py
data = pd.read_csv("./Test.csv")
```

資料預處理

```py
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emojis).replace('-', '')
    return text

data['text'] = data['text'].apply(preprocessor)
```

將句子分開為單字，並做詞幹提取(stemming)的動作。

```py

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
```

TF-IDF 特徵，會使用到前面建立的預處理函數 `preprocessor()`

```py
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None,
                        tokenizer=tokenizer_porter, use_idf=True, norm='l2', smooth_idf=True)
y = data.label.values
x = tfidf.fit_transform(data.text)
```

將資料集切割為訓練資料和測試資料

```py
X_train, X_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size=0.5, shuffle=False)
```

建立邏輯斯回歸的模型(Logistic Regression)，顯示準確率

```py
clf = LogisticRegressionCV(cv=6, scoring='accuracy', random_state=0,
                           n_jobs=-1, verbose=3, max_iter=500).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
```

## 執行程式

安裝相依套件

```
$ pip install -r requirements.txt
```

執行主程式

```
$ python main.py
```

## 參考資料

- [機器學習應用-「垃圾訊息偵測」與「TF-IDF 介紹」(含範例程式)](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E6%87%89%E7%94%A8-%E5%9E%83%E5%9C%BE%E8%A8%8A%E6%81%AF%E5%81%B5%E6%B8%AC-%E8%88%87-tf-idf%E4%BB%8B%E7%B4%B9-%E5%90%AB%E7%AF%84%E4%BE%8B%E7%A8%8B%E5%BC%8F-2cddc7f7b2c5)
- [Python 深度學習筆記(五)：使用 NLTK 進行自然語言處理](https://yanwei-liu.medium.com/python%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-%E4%BA%94-%E4%BD%BF%E7%94%A8nltk%E9%80%B2%E8%A1%8C%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86-24fba36f3896)
- [文本数据预处理：sklearn 中 CountVectorizer、TfidfTransformer 和 TfidfVectorizer](https://blog.csdn.net/m0_37324740/article/details/79411651)
