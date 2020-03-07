import numpy as np

# 행렬 = 2차원 배열
A = np.array([[1, 2], [3, 4], [5, 6]]) # (3, 2)
B = np.array([[4, 5, 6]]) # (1, 3)

# 벡터 = 1차원 배열
x = np.array([1, 2]) # (2, )
y = np.array([3, 4, 5]) # (3, )

print("-- np.dot --")
print("A matrix: ", np.ndim(A), A.shape)
print("B matrix: ", np.ndim(B), B.shape)

print("x vector: ", np.ndim(x), x.shape)
print("y vector: ", np.ndim(y), y.shape)

print("B*A: ", np.dot(B, A)) # (1, 3) * (3, 2) = (1, 2)

print("A*x: ", np.dot(A, x)) # (3, 2) * (2, ) = (3, )
print("y*A: ", np.dot(y, A)) # (3, ) * (3, 2) = (2, )

print("B*y: ", np.dot(B, y)) # (1, 3) * (3, ) = (1, )

# argmax() 는 최댓값의 인데스를 가져온다. axis는 차원을 의미하며 0부터 시작한다.
x = np.array([[[1, 8, 1, 0], [2, 1, 4, 3]], [[1, 6, 1, 2], [0, 10, 0, 0]], [[0, 1, 0, 9], [7, 0, 0, 3]]])

print("\n-- np.argmax --")
print("x shape: ", x.shape)
print(x)

y0 = np.argmax(x, axis=0)
print("x argmax(axis=0), shape: ", y0.shape)
print(y0)

y1 = np.argmax(x, axis=1)
print("x argmax(axis=1), shape: ", y1.shape)
print(y1)

y2 = np.argmax(x, axis=2)
print("x argmax(axis=2), shape: ", y2.shape)
print(y2)

# 배열 == 연산자, sum()
y = np.array([[1, 2, 1, 0], [1, 1, 1, 1]])
t = np.array([[1, 2, 0, 0], [1, 1, 1, 1]])

print("\n-- np.sum --")
print(y==t)
print(np.sum(y==t))