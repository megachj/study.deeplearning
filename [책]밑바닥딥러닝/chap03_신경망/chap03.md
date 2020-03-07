# 신경망
## 퍼셉트론에서 신경망으로
> 신경망은 다층 뉴런(노드)으로 구성된 것을 말한다.
입력층(0층), 은닉층(1층, ..., n-1층), 출력층(n층)이라고 하면 총 n층 신경망이라고 한다.

퍼셉트론의 출력층에서는 가중치와 입력값, 편향의 식이 0이하이면 0, 0초과이면 1을 출력했다. 이때 이 식을 활성화 함수로 표현할 수 있다.

이처럼 입력 신호의 총합을 출력 신호로 변환하는 함수를 일반적으로 활성화 함수(activation function)라 한다.

이를 뉴런으로 표현하면 다음과 같다.
![](https://images.velog.io/images/megachj/post/3133ddae-19d6-4221-808d-79009de87ad2/unnamed.png)

자세히 보면 출력층 뉴런은 두 뉴런이 합쳐진 것이다. 가중치 신호를 조합한 a뉴런과 활성화 함수 h() 를 통과한 y 뉴런이다. 보통은 그냥 원으로 뉴런을 그리지만 신경망 동작을 명확히 드러내고자 할 땐 이렇게 활성화 처리 과정을 명시하기도 한다.

## 활성화 함수
![](https://images.velog.io/images/megachj/post/48405e53-fd19-4f8d-81b2-916df47553ed/49909393-11c47b80-fec2-11e8-8fcd-d9d54b8b0258.png)

처음 퍼셉트론에서 살펴본, 임계값을 기준으로 출력이 바뀌는 함수를 계단 함수라고 하고, 이것도 사실 활성화 함수이다. 계단식은 신경망에서 활성화 함수로 사용하기에 단점들이 있어서 신경망에서는 위 그림처럼 다양한 신경망을 사용한다.

### 몇 가지 활성화 함수 구현
```python
import numpy as np
import matplotlib.pylab as plt

# 계단 함수
def step_function(x):
	return np.array(x > 0, dtype=np.int)

# 시그모이드 함수
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# ReLU 함수
def relu(x):
	return np.maximum(0, x)

# 메인
x = np.arange(-5,0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```

## 다차원 배열의 계산
```python
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]]) # 2차원 배열
B = np.array([7, 8]) # 1차원 배열
C = np.array([7, 8, 9]) # 1차원 배열

np.ndim(A) # 2, 2차원 배열
A.shape # (3, 2)
B.shape # (2, )
C.shape # (3, )

# 행렬 곱은 순서에따라 차원이 맞아야 한다.
np.dot(A, B) # 3*2 와 2*1 이라 맞음
np.dot(C, A) # 1*3 와 3*2 이라 맞음
# np.dot(A, C) 3*2 와 3*1 이라 맞지 않음
```

## 출력층 설계하기
기계학습 문제는 다음 2가지로 분류된다.
* 분류(Classification): 데이터가 어느 클래스에 속하는냐는 문제
* 회귀(Regression): 입력 데이터에서 (연속적인) 수치를 예측하는 문제

### 항등 함수와 소프트맥스 함수
항등 함수는 입력을 그대로 출력하는 함수이다. f(x) = x.

소프트맥스 함수는 분류 문제에서 사용하는데, 식은 다음과 같다.
![](https://images.velog.io/images/megachj/post/7a21465b-f5e4-41a3-aed3-47b152237beb/img.png)

식에서 본 것처럼 실제로 소프트맥스 함수를 구현한다면 지수 함수여서 오버플로가 발생할 수 있다. 그래서 다음처럼 수식을 개선해, 오버플로를 피할 수 있다.
![](https://images.velog.io/images/megachj/post/810b00b3-6805-4adf-9f96-829e1b3f4ee3/Internet_20200301_105641.png)

지수 함수는 단조 증가 함수이므로, 소프트맥스 함수를 적용해도 각 원소의 대소 관계는 변하지 않는다. 그래서 신경망을 이용한 분류에서 학습이 아닌 추론 과정에서는 마지막 출력층의 소프트맥스 함수를 생략해도 된다.
* 기계학습의 문제 풀이는 학습과 추론 두 단계로 이루어지는데, 학습은 모델 학습 과정을 의미하고, 추론은 학습된 모델로 미지의 데이터를 추론을 수행하는 것을 말한다. 

## 손글씨 숫자 인식
MNIST 데이터셋을 이용해 손글씨 숫자 인식 예제를 진행해보자.