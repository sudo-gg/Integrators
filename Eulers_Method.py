from unittest import result
import matplotlib.pyplot as plt
from numpy import linspace

def func(x,y):
    # our dy/dx function
    # for example, dy/dx = 2x**2 - 1
    # this has y = (2/3)x**3 - x + C
    return 2*x*2-1


def EulersMethod1V(f: func, x_0,y_0, x_n, dx):
    result = []
    x_list = []
    y_list = []
    x=x_0
    y = y_0
    while x < x_n:
        y += dx * f(x,y) + 0.5 *dx**2 * f(x,y)  # 2nd order Euler's method
        x += dx
        x_list.append(x)
        y_list.append(y)
    result.append(x_list)
    result.append(y_list)
    return result

def RK2(f: func, x_0,y_0, x_n, dx):
    result = []
    x_list = []
    y_list = []
    x = x_0
    y = y_0
    while x < x_n:
        k1 = f(x,y)
        k2 = f(x + dx/2, y + dx * k1 / 2)
        x_list.append(x)
        y_list.append(y)
        y += dx * (k1 + k2) / 2
        x += dx
    result.append(x_list)
    result.append(y_list)
    return result

def RK4(f: func, x_0,y_0, x_n, dx):
    result = []
    x_list = []
    y_list = []
    x = x_0
    y = y_0
    while x < x_n:
        k1 = f(x, y)
        k2 = f(x + dx / 2, y + dx * k1 / 2)
        k3 = f(x + dx / 2, y + dx * k2 / 2)
        k4 = f(x + dx, y + dx * k3)
        x_list.append(x)
        y_list.append(y)
        y += dx * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += dx
    result.append(x_list)
    result.append(y_list)
    return result




if __name__ == "__main__":
    EulersM1V = EulersMethod1V(func, -10, 2*(-10)**3/3 - 10, 10, 0.1)
    RK2V = RK2(func, -10,2*(-10)**3/3 - 10, 10, 0.1)
    RK4V = RK4(func, -10, 2*(-10)**3/3 - 10, 10, 0.1)
    graph = plt.subplot()
    plt.plot(EulersM1V[0], EulersM1V[1], label='Euler\'s Method')
    plt.plot(RK2V[0], RK2V[1], label='RK2 Method')
    plt.plot(RK4V[0], RK4V[1], label='RK4 Method')
    plt.title('Comparison of Numerical Methods')
    plt.plot(linspace(-10, 10, 100), (2/3) * linspace(-10, 10, 100)**3 - linspace(-10, 10, 100), label='Exact Solution')
    plt.legend()
    plt.show()