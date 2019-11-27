#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import random
from keras.layers import Input,Embedding,Dot,Reshape,Dense
from keras.models import Model


# In[26]:


data=pd.read_csv('C:/Users/HOME/Desktop/8차/통계학 세미나/팀플/잠재요인 돌릴때 쓰는 데이터/score.csv')


# In[27]:


data.head()


# In[28]:


data=data.fillna(0)


# In[29]:


#일단 딥러닝 데이터가 많은 것에 적합하니까, 결측값 0으로 채움


# In[30]:


#신경망 임베딩 : 이산형 변수를 연속벡터로 나타내는 방법임.
#임베딩 공간에 서로 유사한 실체를 가깝게 배치 한다는 뜻임.


# In[31]:


col=['beer_id','username','score']


# In[32]:


data2 = data[col]


# In[33]:


data2.head()


# In[34]:


#Supervised Learning task 에 대한 부분인데 적용을 못하겠음.

#beer_id와 beer의 고유 인덱스 index 맵핑
#그러니까 신경망에는 정수만 입력할 수 있기 때문에, 각 고유 beer_id에서 정수로의 매핑을 해야함...
#score 에도 동일한 작업을 해줘야 하는데..


# In[61]:


print(data2['beer_id'][1:10])
print(data2['score'][1:10])
print(data2['username'][1:10])
# 각각을 정수로 매핑해야함..


# In[52]:


Beer_ID=data2['beer_id']
Score = data2['score']
User = data2['username']
print(Beer_ID[1:5])
print(Score[1:5])
print(User[1:5])
#신경망에서 인식할수 있게 각각을 정수로 매팽해야 함


# In[74]:


len(Beer_ID.unique())
#309542명의 맥주 종류.


# In[76]:


len(User.unique())
#163935명의 평가자


# In[ ]:


#embedding neural network에서 임베딩의 목표는 손실을 최소화 하기 위해서 훈련 중에 조정되는 
#신경망의 가중치(weight이다.) 신경망은 1과 0사이의 예측을 출력하며, 이는 실제값과 비교된다.
# 이모델은 Adam Opitimizer(확률적 경사 하강법 의 변형)
#로  컴파일 되며 이진 분류 문제에 대한 binary_crossentropy 값을 최소화


# In[15]:


def beer_embedding_model(embedding_size = 50, classification=False):
    
    #1차원의 입력
    beer = input(name='beer',shape=[1])
    score = input(name = 'link',shape=[1])
    
    #beer 임베딩 (shape wil be (None, 1, 50))
    beer_embedding = Embedding(name = 'beer_embedding',
                              input_dim =len(beer_index),
                              output_dim = embedding_size)(book)
    
    #score 임베딩(shape wil be (None, 1, 50))
    score_embedding = Embedding(name ='score_embedding',
                                input_dim = len(score_index),
                                output_dim = embedding_size)(score)
    
    #내적으로 맥주 임배딩과 score 임베딩을 한개의 임베딩 벡터로 변형
    #shape will be (None , 1,1)
    
    merged = Dot(name = 'dot_product',normalize = True,
                axes=2)([beer_embedding,socre_embedding])
    
    #단일 숫자로 shape 변형 (shape will be (None,1))
    
    #분류를 위한 결과값 출력
    out = Dense(1,activation = 'sigmoid')(merged)
    model = Model(inputs = [book,link],outputs=out)
    
    #원하는 optimizer와 loss 함수로 모델 학습 시작
    model.compile(optimizer='Adam',loss = 'binary_crossemtropy')
    metrics = ['accuracy']
    
    return model
    


# In[ ]:


#학습 샘플 만들기(generating training samples)

#신경망은 batch learner이다. 왜냐하면 한번에 작은 하나의 표본 집합(관찰)을 통해 epoch라고 불리는 많은
#라운드에서 훈련을 받기 떄문. 신경망을 훈련시키는일반적인 방법은 generator을 이용하는것
#전체 결과가 메모리에 저장되지 않도록, yield(not returns)하는 기능. 


# In[22]:


random.seed(100)

def generate_batch(pairs, n_positive = 50, negative_ratio=1.0) :
    
    #트레이닝을 위한 샘플의 batches를 생성하자.
    #batch를 저장할 numpy 배열을 준비하자.
    
    batch_size = n_positive*(1+negative_ratio)
    batch = np.zeros((batch_size,3))
    
    while True:
        #랜덤으로 True인 샘플을 준비한다.
        for idx,(beer_id,score_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx,:] = (book_id,link_id,1)
        idx +=1
        
        #배치 사이즈가 다 찰 때까지 False인 샘플을 추가합니다.
        
        while idx < batch_dize:
            
            #random selection
            random_book = random.randrange(len(beer))
            random_score = random.randrange(len(links))
            
            #True인 샘플이 아니라는 것(False인 샘플 이라는 것)을 체크하자
            if (random_beer,random_score) not in pairs_set:
                
                #False인 샘플을 배치에 추가합니다.
                batch[idx,:] = (random_book,random_link,neg_label)
                idx +=1
                
            #배치에 저장된 데이터들의 순서를 섞습니다.
            np.random.shuffle(batch)
            yield {'beer':batch[:0],'score':batch[:1]}, batch[:,2]
            


# In[ ]:


#generator로 next를 부를때마다, 새 학습 데이터 배치를 가져온다.


# In[ ]:


#early stopping 을 구현하기 위해 validation셋을 사용하지 않기 때문에, trai
n_positive = 1024

gen = generate_batch(pairs, n_positive,negative_ratio=2)

#train
h = model.fit_generator(gen,epochs=15, steps_per_epoch=len(pairs)//n_positive)


# In[ ]:


#임베딩 벡터 추출하기
beer_layer = model.get_layer('beer_embedding')
score_weights = beer_layer.get_weights()[0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




