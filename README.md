# RecSys 공부

Various top-N recommendation systems under the strictly strong generalization.

train, test를 strong generalization + timestamp에 따라 나눔

## Libraries
  - Data Handling
    - pandas
    - numpy array
    - 가끔씩 scipy.sparse
  - Modeling
    - tensorflow.keras
    - numpy
    - RBM은 순수 tensorflow 2.0으로 구현하고 있음
    - item2vec은 처음에는 gensim으로 구현했으나 나중에는 tensorflow로 구현할 예정

## Models
  - ItemPop
     - user들과 가장 interacting이 높았던 item들을 추천
     - 추천시스템의 baseline이며, user의 선호 정도를 분석하고 연구할 수 있음
  - KNN
    - user-based CF을 바탕으로 구현하였으며, query user과 가장 유사한 user들의 item을 추천
    - 다양한 metric을 선형대수적으로 계산하여 적용해 봄
  - EASE
    - item-based CF를 바탕으로 구현하였으며, 왠만한 아래의 AE모델들 보다 accuracy level이 높음
    - 반면 데이터가 큰 경우, 메모리나 시간에 있어서 패널티가 발생할 것이라 생각됨
    - item similarity matrix에 대해 최적화 문제를 풀어 추천
    - 모델명에는 AutoEncoder의 의미가 들어가있으나 사실상 SLIM의 상위호환 모델에 가까움
    - SLIM 모델과 달리 l1 regularization과 학습 파라메터의 positivity 조건이 없음
  - AutoRec
    - AutoEncoder를 이용해 rating을 복구하는 개념
    - 논문의 old한 딥러닝 기법들을 최신화하면 성능이 더 올라가기는 함
    - 생각보다 논문이 불친절하다는 느낌이 들었음
    - U-AutoRec을 implicit + top-N 추천 구조로 바꿔서 구현함
    - I-AutoRec을 top-N 추천 구조로 어떻게 만들어야 할지 감이 안잡힘
  - DeepRec
    - AutoRec을 조금 더 깊게 구현한 버전
    - re-feeding의 개념을 적용하였는데, 여기까지 구현한 코드를 못 봄
  - CDAE
    - AutoRec의 목적을 돌이켜보자면, inference 단계에서 unseen data의 rating을 복구하는 것이므로 query는 오염(corruption)되어 있다고 봐야함
    - 따라서 학습단계에서도 input data를 오염시켜 모델을 강건하게 만들어야 한다는 내용(= Denoising)
    - 간단하게 Dropout만 넣어도 되긴하는데 논문에선 negative sampling까지 언급하고 있어서 feeding 단계를 새로 구현함
    - 일단은 tf.keras로 만들었는데, 나중에는 tensorflow로만 구현할 예정
  - Mult-VAE
    - 변분 추론(variational inference)를 이용하여 latent factor 모델이었던 AE를 latent variable 모델로 바꿈
    - 여기서는 rating을 복구하는 것이 아닌, 한정된 확률 공간 안에서 item이 등장할 확률을 경쟁시킴
    - ML-1M을 써서 그런지 Mult-DAE가 더 성능이 좋더라
  - NCF
    - user와 item을 같은 공간(더 괜찮은 단어가 있었는데 까먹음)에 embedding하여 similarity를 계산하자는 것이 주된 개념
    - 저자는 MLP를 이용하여 inner product 연산을 근사할 수 있다고 주장함(근데 이 내용이 이후에 많이 까임)
    - 구현이나 학습은 쉽지만 I-AutoRec과 마찬가지로 top-N 구조로 어떻게 만들어야 할지 감이 안잡힘
    - inference 단계에서 user query가 주어지면, 각각의 item과 pairing하여 다 넣어봐야 하므로 시간이 오래걸림
    - 따라서 논문은 item를 랜덤하게 뽑아서 query와 함께 infer한다고 함(AkNN 비슷한 알고리즘을 적용하면 어떨까 싶음)
  - Item2Vec
    - word2vec과 같은 개념으로, 두 item이 같은 user에 의해 interact된 경우 비슷한 embedding을 갖도록 학습
    - gensim으로 구현하면 top-N 추천에 굉장히 편리함
    - tf로 구현하면 시간이 더 절약될까 싶어 시도 중
  - RBM
    - AE의 구버전
    - tf.keras로 구현키가 번거로워 tf만 사용하기로 함
  - HierTCN
    - session-based = hierarchical * (GRU + TCN)
    - item의 hidden representative를 이용함(side info 이용가능)
  - AkNN
    - 정확도를 약간 희생해서 속도를 올린 KNN
---\
### not yet implemented
  - RaCT
    - mult-VAE에 강화학습의 actor-critic를 적용하여 성능을 올린 모델
  - RecVAE
    - VAE에 multinomial 안쓰고 cross entropy씀 + 좀 더 두꺼운 모델
  - H+Vamp
    - flex한 prior사용
  - KGNN
    - knowledge graph!
  - GraphSAGE
    - item의 hidden representative를 학습하는데 있어서 GNN을 사용
    - neighbor의 representative와 자신의 representative를 aggregation해서 업데이트

## Losses
  - BRP
  - Lambdarank
