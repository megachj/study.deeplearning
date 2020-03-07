import numpy as np

# 행렬 = 2차원 배열
A = np.array([[1, 2], [3, 4], [5, 6]]) # (3, 2)
B = np.array([[4, 5, 6]]) # (1, 3)

# 벡터 = 1차원 배열
x = np.array([1, 2]) # (2, )
y = np.array([3, 4, 5]) # (3, )

print("A 행렬: ", np.ndim(A), A.shape)
print("B 행렬: ", np.ndim(B), B.shape)

print("x 벡터: ", np.ndim(x), x.shape)
print("y 벡터: ", np.ndim(y), y.shape)

print("B*A: ", np.dot(B, A)) # (1, 3) * (3, 2) = (1, 2)

print("A*x: ", np.dot(A, x)) # (3, 2) * (2, ) = (3, )
print("y*A: ", np.dot(y, A)) # (3, ) * (3, 2) = (2, )

print("B*y: ", np.dot(B, y)) # (1, 3) * (3, ) = (1, )