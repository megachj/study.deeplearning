import sys, os
import numpy as np

# 1단계 상위 폴더 경로 추가, ../
# n단계 상위인 경우 폴더 경로는 `os.path.dirname(os.path.abspath(` 가 1단계 상위를 의미하므로 n개 겹치면 된다.
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}

        ## randn(m, n): 표준정규분포(평균 0, 표준편차 1)를 따르는 행렬 m*n 생성
        ## 여기에 weight_init_std 를 곱하니 평균은 0, 표준편차는 weight_init_std 인 정규분포가 된다.
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x: 입력 데이터, t: 정답 레이블
    def loss(self, x, t):
        # 순전파 결과
        y = self.predict(x)

        # 손실 함수 계산
        e = cross_entropy_error(y, t)
        
        print('loss: ', x.shape, t.shape, y.shape, e)

        return e

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x: 입력 데이터, t: 정답 레이블
    def numerical_gradient(self, x, t):
        print('numerical_gradient: ', x.shape, t.shape)

        # 손실 함수
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        print('grad W1 start...')
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        print('grad W1 end: ', grads['W1'].shape)

        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        print('grad b1 end: ', grads['b1'].shape)

        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        print('grad W2 end: ', grads['W2'].shape)

        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        print('grad b2 end: ', grads['b2'].shape)

        return grads