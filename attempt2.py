import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, odeint
import libs.LeeMaths as lm

VeffEq = lm.exp("-(x**2+y**2)/2 - (1-u)/(((x+u)**2+y**2)**(1/2))-u/(((x+u-1)**2+y**2)**(1/2))")
u = 2

def Veff(x, y, u):
    return VeffEq.evaluate(dict(x=x, y=y, u=u)).__float__()

def VeffHard(x, y, u):
    a = -(x**2+y**2)/2
    b = - (1-u)/(np.sqrt((x+u)**2+y**2))
    c = - u/(np.sqrt((x+u-1)**2+y**2))
    return a+b+c

def f(state, t):
    x, y, v_x, v_y = state  # unpack state vector
    global u

    xdot = v_x
    ydot = v_y

    delta = VeffHard(x + xdot, y + ydot, u) - VeffHard(x, y, u)
    vxdot = - (delta / xdot) + (2 * v_y)
    vydot = - (delta / ydot) - (2 * v_x)
    return xdot, ydot, vxdot, vydot

t = np.arange(0.0, 1, 0.001) # time steps
fig, ax = plt.subplots()

ax.grid()
plt.axvline(0, color='black')
plt.axhline(0, color='black')

initial1=[0, -2, -4, 0.1]
states1 = odeint(f, initial1, t)
#states1 = solve_ivp(fun=lambda t, state: f(state, t), t_span=(0, 1), t_eval=tt, y0=initial1)
flow1=ax.plot(states1[:,0],states1[:,1],color='royalblue')[0]

plt.plot(-u, 0,'ro')
plt.plot(1 - u, 0,'ro')

plt.show()