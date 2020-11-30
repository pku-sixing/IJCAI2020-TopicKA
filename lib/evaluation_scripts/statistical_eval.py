import math
def factorial(n):
    return math.factorial(n)

def b(x, n, p=0.5):
    tmp = 1.0
    for y in range(x+1,n+1):
        tmp *= y / (y-x)
    return tmp *  np.power(0.5, n)
#     return factorial(n)/factorial(n-x)/factorial(x) * np.power(0.5, n)

def B(x, n,p=0.5):
    # p = 0.5
    tmp = 0
    for y in range(0, x+1):
        tmp += b(y,n, p)
    return tmp

def sign_test(win, loss):
    return B(loss, win+loss)
