# coding: utf-8
import sys, os
import numpy as np
from PIL import Image

# 1단계 상위 폴더 경로 추가, ../
# n단계 상위인 경우 폴더 경로는 `os.path.dirname(os.path.abspath(` 가 1단계 상위를 의미하므로 n개 겹치면 된다.
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.mnist import load_mnist

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# mnist 데이터 np 배열로 가져오기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 데이터 형상 출력
print("-- print data shape --")
print("x_train: ", x_train.shape) # (60000, 784)
print("t_train: ", t_train.shape) # (60000, )
print("x_test: ", x_test.shape) # (10000, 784)
print("t_test: ", t_test.shape) # (10000, )

# 이미지 한개 출력
img = x_train[0].reshape(28, 28) # (784, ) 이므로 이미지 출력을 위해 (28, 28) 로 재변형
label = t_train[0]
img_show(img)