# Beer-Recommendation-System
2019-2학기 '통계학세미나' 수업에서 수행한 팀 프로젝트입니다.  
이 프로젝트는 개인에게 맞춤형 맥주를 추천해줄 수 있는 __맥주 추천 알고리즘__ 개발 프로젝트입니다.  
발표 자료는 [여기에서](https://github.com/statseminar/Beer-recommendation-system/blob/master/2019-2_%EB%A7%A5%EC%A3%BC_%EC%B5%9C%EC%A2%85%EB%B0%9C%ED%91%9C(%EC%99%84%EC%84%B1%EB%B3%B8).pdf) 다운받을 수 있습니다.

---
### 수업 정보
- 2019년 2학기 (2019.09.02-2019.12.21)
- 통계학 세미나
  - 전공(응용통계학)
  - 임창원 교수님
### 분석 개요
- 사용한 데이터는 [Kaggle의 Beers, Breweries, and Beer Reviews](https://www.kaggle.com/ehallmar/beers-breweries-and-beer-reviews)로, 그 중 `reviews.csv` (약 900만 개 데이터)를 추천 알고리즘을 개발하는 데에 사용 
- 맥주시장의 현황 시각화
- 평가지표: RMSE
### Part 1. 머신러닝
- 협업 필터링(Collaborative Filtering)을 기반으로 한 최근접 이웃 방식- IBCF(Item-Based Collaborative Filtering), 잠재요인 방식의 SVD, NMF 알고리즘 사용 
- python 라이브러리 중 하나인 `SURPRISE` 패키지 활용
- 데이터가 900만 개로 너무 많아 머신러닝 알고리즘으로는 수행 시간이 너무 오래 걸려 많은 전처리 끝에 약 10만개의 데이터만을 이용해 수행하였음
### Part 2. 딥러닝(Keras)
- 딥러닝은 온전히 900만 개의 데이터를 모두 사용하여 좀 더 믿을 수 있는 성능을 도출할 수 있었음
- Layer의 개수를 추가, activation 다양화, optimizer 다양화로 많은 모델을 만들어 보고, 가장 좋은 성능을 내는 모델 도출 
### 결론 
- 가장 좋은 성능은 머신러닝 기법의 NMF
- 다양한 추천 알고리즘을 실제로 경험해볼 수 있어 의미있는 프로젝트였지만, 더 좋은 성능을 내기 위해 하이퍼 파라미터 튜닝이나 과적합 규제 방법을 썼으면 하는 아쉬움이 남음.
