import sys, os
import numpy as np
import time

# 1단계 상위 폴더 경로 추가, ../
# n단계 상위인 경우 폴더 경로는 `os.path.dirname(os.path.abspath(` 가 1단계 상위를 의미하므로 n개 겹치면 된다.
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

print('----- start, train mini batch -----\n')
startTime = time.time()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print('Total data. (x_train, t_train), (x_test, t_test): ', x_train.shape, t_train.shape, x_test.shape, t_test.shape)

train_loss_list = []

# 하이퍼파라미터
iters_num = 1 # 반복 횟수
train_size = x_train.shape[0] # 60000 개 데이터
batch_size = 1 # 미니배치 크기
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 총 훈련 데이터 수: batch_size * iters_num
for i in range(iters_num):
    # 미니배치 획득
    print('[Start get batch]...')
    batch_mask = np.random.choice(train_size, batch_size) # (batch_size, ): [0, 60000) 중 100개 랜덤 선택
    x_batch = x_train[batch_mask] # (batch_size, 784), ndim = 2
    t_batch = t_train[batch_mask] # (batch_size, 10), ndim = 2
    print('[Done get batch]...')

    # 기울기 계산
    print('[Start get gradient]...')
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 성능 개선판
    print('[Done get gradient]...')

    # 매개변수 갱신
    print('[Start update network params]...')
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    print('[Done update network params]...')

    # 학습 경과 기록
    print('[Start store loss]...')
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print('[Done store loss]... ', loss)

endTime = time.time() - startTime
print('\n----- end, elapsed ' + str(round(endTime, 2)) + '(s) -----')