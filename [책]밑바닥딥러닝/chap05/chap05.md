## 계산 그래프
계산 그래프(computational graph)는 계산 과정을 그래프로 나타낸 것이다. 노드는 함수(연산)을 나타내고, 에지는 값을 나타낸다.

또 계산 그래프를 이용하면 역전파를 통해 '미분'을 효율적으로 계산할 수 있다.
순전파 방향은 계산 과정이고, 역전파 방향은 미분 과정이다.
![](https://images.velog.io/images/megachj/post/2799b8c2-926c-4e84-8a5a-a820c7168f66/5-1.png)

## 연쇄법칙
노드(함수)가 1개인 계산 그래프의 역전파는 다음과 같다.

![](https://images.velog.io/images/megachj/post/adda7bf1-8f32-46ac-886f-8564df4e9161/5-2.png)

연결된 노드(함수)가 여러개이면 합성함수를 나타내고, 이때 계산 그래프의 역전파를 그리면 다음과 같다.

![](https://images.velog.io/images/megachj/post/8ae9b2fa-f691-4c6e-a354-159effff2285/5-3.png)

위 그림처럼 여러 연결된 노드의 역전파는 합성함수의 미분과 동일하다.

## 역전파
덧셈과 곱셈의 연산에 대한 역전파를 살펴보자.

![](https://images.velog.io/images/megachj/post/7c6aba31-eaf1-4f87-b1c2-2a79a6f82a59/5-4.png)

## 단순한 계층 구현
책 예제를 바탕으로 앞에서 살펴본 덧셈, 곱셈 계층을 구현해보자.

```python
# 곱셈 계층
class MulLayer:
    def __init__(self):
    	self.x = None
        self.y = None
        
    def forward(self, x, y):
    	self.x = x
        self.y = y       
        return x * y
   
    def backward(self, dout):
    	dx = dout * self.y
        dy = dout * self.x    
        return dx, dy

# 덧셈 계층
class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
```

## 활성화 함수 계층 구현
### ReLU 계층
ReLU 계층의 계산 그래프를 살펴보자.

![](https://images.velog.io/images/megachj/post/79951fbf-38f1-4b3b-a31f-e596ca708689/5-5.png)

ReLU 계층은 다음처럼 구현할 수 있다.
```python
class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
        
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
```

### Sigmoid 계층
Sigmoid 계층의 계산 그래프를 살펴보자.

![](https://images.velog.io/images/megachj/post/222ed97a-d9b1-4fbb-9708-6af0a60fe620/5-6.png)

Sigmoid 계층은 다음처럼 구현할 수 있다.
```python
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
```

## Affine/Softmax 계층 구현
### Affine 계층
> 신경망의 순전파 때 수행하는 행렬의 곱은 기하학에서는 어파인 변환(affine transformation)이라고 한다. 이 책에서는 어파인 변환을 수행하는 처리를 `Affine 계층`이라는 이름으로 구현한다.

Affine 계층의 계산 그래프를 살펴보자.

![](https://images.velog.io/images/megachj/post/a9193667-63de-4832-bda6-0d5b65ec09b4/5-8.png)

1번을 구하기 위해서 벡터를 벡터로 미분하는 것을 살펴보자.

![](https://images.velog.io/images/megachj/post/aa77c0f9-2d37-4da3-9981-245d829edead/5-7.png)

그러면 1번, 2번을 구해보자.

![](https://images.velog.io/images/megachj/post/b21f053b-ec66-4008-8c96-b4023c061a1e/5-9.png)

![](https://images.velog.io/images/megachj/post/0f5725fd-e049-4f98-8f9b-20a164111471/5-10.png)

### 배치용 Affine 계층

![](https://images.velog.io/images/megachj/post/24e726ad-5795-4998-9b24-a324afa0f1f4/5-11.png)

Affine 계층은 다음처럼 구현할 수 있다.

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
        
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
```

### Softmax-with-Loss 계층
소프트맥스 계층과 손실 함수인 교차 엔트로피 오차를 합해서 `Softmax-with-Loss 계층` 이라고 한다.

![](https://images.velog.io/images/megachj/post/117981ac-da56-4d6c-94d7-eb237effb03e/5-12.png)

![](https://images.velog.io/images/megachj/post/21807c85-ceaa-416e-961d-b6df1e97b6e5/5-13.png)

Softmax-with-Loss 계층은 다음처럼 구현할 수 있다.

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실
        self.y = None # softmax 출력
        slef.t = None # 정답 레이블(원-핫 벡터)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
        
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
```

## 오차역전파법 구현
책 예제를 바탕으로 앞에서 구현한 계층들을 조합해 신경망을 구축해보자.