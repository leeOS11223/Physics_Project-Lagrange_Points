import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, odeint
import libs.LeeMaths as lm

VeffEq = lm.exp("-(x**2+y**2)/2 - (1-u)/(((x+u)**2+y**2)**(1/2))-u/(((x+u-1)**2+y**2)**(1/2))")

m1 = 5.972 * 10**24
m2 = 7.34767309 * 10**22

u = m2/(m1+m2)


VeffEqdx = VeffEq.diff()
VeffEqdy = VeffEq.diff(lm.exp("y"))

print(VeffEq.diff().ugly())
print(VeffEq.diff(lm.exp("y")).ugly())

def Veff(x, y, u):
    return VeffEq.evaluate(dict(x=x, y=y, u=u)).__float__()

def Veffdx(x, y, u):
    return VeffEqdx.evaluate(dict(x=x, y=y, u=u)).__float__()

def Veffdy(x, y, u):
    return VeffEqdy.evaluate(dict(x=x, y=y, u=u)).__float__()

def VeffdxHard(x, y, u):
    return ((1-u)*(x+u))/((x+u)**2+y**2)**(3/2)+(u*(x+u-1))/((x+u-1)**2+y**2)**(3/2)-x


def VeffdyHard(x, y, u):
    return ((1-u)*y)/(y**2+(x+u)**2)**(3/2)+(u*y)/(y**2+(x+u-1)**2)**(3/2)-y


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

    #delta = VeffHard(x + xdot, y + ydot, u) - VeffHard(x, y, u)
    #vxdot = - (delta / xdot) + (2 * v_y)
    #vydot = - (delta / ydot) - (2 * v_x)

    vxdot = - VeffdxHard(x, y, u) + (2 * v_y)
    vydot = - VeffdyHard(x, y, u) - (2 * v_x)
    return xdot, ydot, vxdot, vydot

t = np.arange(0.0, 10, 0.001) # time steps
fig, ax = plt.subplots()

ax.grid()
plt.axvline(0, color='black')
plt.axhline(0, color='black')

initial1=[0.8, 0, 0.1, 0.1]
states1 = odeint(f, initial1, t)
flow1=ax.plot(states1[:,0],states1[:,1],color='royalblue')[0]

# for x in np.arange(-1,1,0.4):
#     for y in np.arange(-1,1,0.4):
#         initial1 = [x, y, 0, 0]
#         states1 = odeint(f, initial1, t)
#         xx=states1[:, 0]
#         yy=states1[:, 1]
#         if(np.sqrt(xx[-1]**2+yy[-1]**2)<1):
#             flow1 = ax.plot(xx, yy)[0]

plt.plot(-u, 0,'ro')
plt.plot(1 - u, 0,'ro')

plt.title("u="+str(u)+", pos=("+str(initial1[0])+", "+str(initial1[1])+"), vel=("+str(initial1[2])+", "+str(initial1[3])+")")

plt.show()