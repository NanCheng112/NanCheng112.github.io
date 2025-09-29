---
title: 统计学习方法-感知机
date: 2025-9-29 15:44:00
updated: 2025-9-29 15:44:00
tags:
    - Python
    - 机器学习
categories:
    - 机器学习
katex: true
---
利用iris数据集中的$sepal length$，$sepal width$作为特征实现感知机算法，首先进行数据的加载与预览。


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>




```python
plt.scatter(df[:50]['sepal length (cm)'], df[:50]['sepal width (cm)'], label='0')
plt.scatter(df[50:100]['sepal length (cm)'], df[50:100]['sepal width (cm)'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
```




    <matplotlib.legend.Legend at 0x174b6471e20>



  
![png](https://github.com/NanCheng112/NanCheng112.github.io/blob/hexo/source/_posts/ml_learning/perceptron_files/perceptron_3_1.png?raw=true)
    



```python
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])
```

### **感知机学习算法的原始形式**

采取**随机梯度下降法**，感知机模型$f(x)=sign(w \cdot x + b)$，选取初值$w_0$与$b_0$，在训练集中随机抽取误分类点使梯度下降。    
$w=w+\eta y_i x_i$    
$b=b+\eta y_i$     
这种算法在直观上解释即当一个实例点被错误分类时，则调整$w$与$b$的值，使分离超平面向该误分类点进行移动。


```python
# 感知机原始形式模型
class Model_original:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1)
        self.b = 0
        self.l_rate = 0.1

    def sign(self, x, w, b):
        y = x @ w + b
        return y
    
    def fit(self, X_train, y_train):
        wrong = False
        while not wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                wrong = True

        return "The perceptron model has been successfully built"
```


```python
perceptron = Model_original()
perceptron.fit(X, y)
```




    'The perceptron model has been successfully built'




```python
x_points = np.linspace(4, 7, 10)
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'o', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'o', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
```




    <matplotlib.legend.Legend at 0x174b68ffa40>




    
![png](https://github.com/NanCheng112/NanCheng112.github.io/blob/hexo/source/_posts/ml_learning/perceptron_files/perceptron_9_1.png?raw=true)
    


### **感知机的学习算法的对偶形式**

同样采取**随机梯度下降法**，感知机模型$f(x)=sign(\sum_{j=1}^N a_j y_j x_j \cdot x + b)$，选取初值$\alpha = 0$与$b = 0$，在训练集中随机抽取误分类点使梯度下降。    
$\alpha_i = \alpha_i + \eta$    
$b=b+\eta y_i$     
对偶形式中训练实例仅以内积的形式出现，故可以预先将训练集中实例间的内积计算出来以矩阵形式存储，该矩阵即为$Gram$矩阵
$$G = [x_i \cdot x_j]_{N \times N}$$


```python
# 计算Gram矩阵
gram = X @ X.T
```


```python
# 感知机对偶形式模型
class Model_dual():
    def __init__(self):
        self.b = 0
        self.l_rate = 0.1
    
    def sign(self, alpha, y, gram_row, b):
        result = np.sum(alpha * y * gram_row) + b
        return result
    
    def fit(self, gram, y_train):
        self.alpha = np.zeros(len(y_train))
        wrong = False
        while not wrong:
            wrong_count = 0
            for d in range(len(gram)):
                gram_row = gram[d]
                y = y_train[d]
                if y * self.sign(self.alpha, y_train, gram_row, self.b) <= 0:
                    self.alpha[d] += self.l_rate
                    self.b += self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                wrong = True

        return "The perceptron model has been successfully built"

```


```python
perceptron_dual = Model_dual()
perceptron_dual.fit(gram, y)
```




    'The perceptron model has been successfully built'




```python
x_points = np.linspace(4, 7, 10)
# 计算权重向量 w = sum(alpha_i * y_i * x_i)
w = np.sum(perceptron_dual.alpha.reshape(-1, 1) * y.reshape(-1, 1) * X, axis=0)
# 决策边界方程: w1*x + w2*y + b = 0 → y = -(w1*x + b)/w2
y_ = -(w[0] * x_points + perceptron_dual.b) / w[1]
plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'o', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'o', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
```




    <matplotlib.legend.Legend at 0x174b68b82f0>




    
![png](https://github.com/NanCheng112/NanCheng112.github.io/blob/hexo/source/_posts/ml_learning/perceptron_files/perceptron_15_1.png?raw=true)
    


当训练集线性可分时，感知机学习算法总是收敛，且存在无穷解，其解由于不同的初值与不同的迭代顺序而可能有所不同
