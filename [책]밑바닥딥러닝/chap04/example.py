import numpy as np

class A:
    def __init__(self):
        self.count = 0
        self.params = {}
        self.params['W1'] = np.random.randn(5, 2)

    def loss(self, x, t):
        return 3.14
        
    
    def ex_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grad = ex_gradient(loss_W, self.params['W1'])

        return grad

net = A()
x = np.random.randn(1, 5)
t = np.random.randn(1, 2)

net.ex_gradient(x, t)