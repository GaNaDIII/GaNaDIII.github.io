---
title: 연속형 - 범주형 컬럼 관계 EDA
date: 2026-01-01
bookmark: true
---

모델을 만들기 전에, EDA를 하면서 분포를 확인하거나, 뭔가 깨달음?을 얻어야 된다

근데 나는 시각화나 그런 거를 별로 중요하게 생각하지 않고 넘어갔기에..(과거의나는진짜뭐지?)

이번 기회에 연습을 해보면서 데이터를 여러 방면으로 EDA해서 모델 만들기에 도움을 줬으면 한다!!

# 데이터 소개

데이터는 간단하게 kaggle의 [spaceship titanic 데이터](https://www.kaggle.com/competitions/spaceship-titanic/data)를 가져왔다

Transported(yes/no)에 관해서 예측하는 것이고, 예측에 사용되는 컬럼들은 크게 ID, 연속형 컬럼들, 범주형 컬럼들 등이 있다. 

데이터를 가져오자마자, head(), info(), columns, describe() 등의 함수로 데이터 분포를 확인하자.

![train_df distribution check](/assets/img/train_df_describe.png)
(describe함수는 연속형 컬럼에 대한 min, max와 같은 값들을 제공해줌)

```python
train_df['Transported'].value_counts(normalize=True)
```

이렇게 value_counts 함수를 사용해서 타깃 컬럼의 분포를 확인한다(극단적으로 치우처져 있으면 데이터 증강 등의 방법을 모색해야 하므로...)
<img src="/assets/img/thumbnail/transported_distribution.png" style="width:50%;">

이거는 연습용 데이터set이라서 고르게 분포된 것을 확인할 수 있다...

# 범주형 타겟 컬럼과 연속형 컬럼의 관계 확인하기

범주형 타겟과 연속형 컬럼(일단 하나..)의 관계를 확인해보자

박스플롯, 바이올린 플롯 등의 시각화 방법을 써서 확인할 수 있다..

나는 무난하게 박스플롯을 써서 시각화해봤다..

## 박스플롯으로 연속형 - 범주형 컬럼 관계 확인하기

```python
train_df.groupby('Transported')['Age'].mean() ##먼저 Transported값으로 사람들을 나눔(yes/no) -> 그리고 나눈 그룹에서 age의 평균을 구함
sns.boxplot(x = 'Transported', y = 'Age', data = train_df) ##박스플롯으로 Transported로 나눈 그룹을 시각화해
```

박스플롯의 좋은 점은 평균 최솟값 최댓값 이상치 모두 한눈에 파악 가능하다는 것이다

<img src="/assets/img/thumbnail/boxplot.png" style='width:50%;'>

이렇게 보면 Transported yes/no 의 그룹에서 그냥저냥 비슷한 것을 볼 수 있다.

## 히스토그램으로 연속형 - 범주형 컬럼 관계 확인하기

연속형이 포함되어 있으니깐 당연히 히스토그램을 쓸 수 있다..

seaborn으로 히스토그램을 그리면 이쁘게 나온다

```python
con_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for c in con_features:
  plt.figure(figsize=(6, 4))
  sns.histplot(data=train_df, x=c, hue='Transported', kde=True, element='step')
  plt.xlim(0, 80)
  plt.show()
```

for문을 돌리면서 연속형 컬럼들을 범주형 타겟 컬럼을 기준으로 히스토그램을 그려 보았다..

<img src='/assets/img/thumbnail/histogram.png' style='width:50%;'>

이렇게 나이 컬럼을 이용해서 히스토그램을 그리면 이쁘게 그려진다

근데....

다른 컬럼들을 이용해서 히스토그램을 그리니깐 xlim을 조정하고 해도 너무 안이쁘게 그려진다...

지피티: 그러면 로그스케일로 변환하죠

그래서 로그 스케일로 변환을 해서 히스토그램을 그린다

왜 로그스케일로 변환하냐? -> 오른쪽이나 왼쪽으로 long tale을 가진 분포를 시각화하면
걍...
<img src='https://www.lgresearch.ai/data/upload/image/blog/1_fcfdf8bb1.png' style='width"50%;'>

이런 식으로 나와서 큰 값 몇개 때문에 분포가 매우 일그러진다..

로그로 변환하면 분포가 펴져서 타깃별 분포 차이가 눈에 들어온다..

(보통 그래프 그릴때 극단적인 값이 있으면 로그 스케일로 변환하듯이...)

```python
#연속형에서 그래프가 이상하니까 로그 변환해서 시각화하자
con_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for c in con_features:
  train_df[c + 'logscale'] = np.log1p(train_df[c])
  plt.figure(figsize=(6, 4))
  sns.histplot(data=train_df, x=c +'logscale', hue='Transported', kde=True, bins=50, element='step')
  plt.xlim(0, 1)
  plt.show()
```
근데 로그는 0이 들어가면 안된다..우리는 음의무한대발산 이런거를 원하지 않는다..

그래서 0을 처리하기 위해서 np.log1p(컬럼) 함수를 이용한다

np.log1p(x)는 log(1 + x) 와 똑같다(밑은 자연로그)

log(1) -> 0 이기 때문에 0을 제대로 0으로 만들기 위해서 1을 더해주는걸로 생각하면 된다..

<img src='/assets/img/thumbnail/logscale.png' style='width:50%;'>

이것도 그렇게 이쁜 분포인거같진않지만 그냥 넘어간다...

## melt 함수를 써서 여러 컬럼들을 행처럼 표현해보자

연속형 컬럼들 보다 보니, 0인 값들이 많아서 0 vs 0 초과 로 구분해서 시각화해보고 싶어졌다..

melt함수를 쓰면 여러 열을 행처럼 사용해서 그래프 그려볼 수 있다고 하니까 이쿠죠!!

```python
#서비스 사용 한 사람 vs 서비스 사용 하지 않은 사람으로 시각화하기
binary_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_df_bin = train_df[binary_cols].gt(0) # -> greater than 0(True/False로 나옴)
train_df_bin = train_df_bin.astype(int) #True는 1로, False는 0으로 변환한다
train_df_bin['Transported'] = train_df['Transported']
```

일단 서비스를 이용한 사람 vs 이용하지 않은 사람을 나누기 위해서 gt(0) 함수를 사용한다

```python
df_long = train_df_bin.melt(id_vars='Transported', var_name='Service', value_name='Used')
sns.countplot(data=df_long, x='Service', hue='Used', palette='Set2')
plt.show()
```
melt함수는 간단하게 컬럼들을 행으로 바꿔준다

<img src='https://i0.wp.com/cmdlinetips.com/wp-content/uploads/2019/06/Pandas_melt_reshape.png?fit=421%2C236&ssl=1' style='width:50%;'>

정확하게는 모르겠는데 그냥 EDA를 할 때, 기준 열을 제외하고 나머지 열들은 다 눕는다고 이해했다....(이렇게전공이해하면 재수강해야됨)

그러면 기준이 되는 열과 다른 열(여기에서는 행이 됨) 을 사용해서 히스토그램을 그릴 수 있다


<img src='/assets/img/thumbnail/meltfunction.png' style='width:50%;'>

이렇게 그림이 그려지는 것을 확인할 수 있다...



다음 포스팅은 범주형 컬럼과 범주형 컬럼의 EDA로 돌아오겠다...

그리고 그 다음 포스팅은 여러 컬럼과 타겟 컬럼의 EDA로 계획함..

이거는 오직 **내가 보려고 정리한거니까** 혹시 보게 되면 이건뭐지...하지말고 스무스하게 넘어가는것을 추천한다...