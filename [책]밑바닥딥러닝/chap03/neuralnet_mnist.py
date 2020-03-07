# coding: utf-8
import sys, os

# 1단계 상위 폴더 경로 추가, ../
# n단계 상위인 경우 폴더 경로는 `os.path.dirname(os.path.abspath(` 가 1단계 상위를 의미하므로 n개 겹치면 된다.
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# 로컬에서 현재 py 파일의 위치를 이상한 곳으로 잡아, 현재 파일 위치로 재조정
# print(os.getcwd())
os.chdir(os.path.dirname(__file__))
# print(os.getcwd())

import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    print("x_test: ", x_test.shape) # 이미지 784짜리 10000개, (10000, 784)
    print("t_test: ", t_test.shape) # 값이 0~9인 레이블 10000개, (10000, )

    return x_test, t_test

def init_network():
    # 학습된 가중치 매개변수 모델
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 네트워크 구조
    ## 입력층: 784개 뉴런
    ## 첫 번째 은닉층: 50개 뉴런, sigmoid
    ## 두 번째 은닉층: 100개 뉴런, sigmoid
    ## 출력층: 10개 뉴런, softmax
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x_test, t_test = get_data() # x_test = (10000, 784), t_test = (10000, )
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) # 각 행에서 최대인 열의 인덱스, 인덱스는 0부터 시작
    accuracy_cnt += np.sum(p == t_test[i:i+batch_size]) # 1차원 배열에서 몇개가 같은지

print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))