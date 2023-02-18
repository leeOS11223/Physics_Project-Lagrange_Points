import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, odeint
import libs.LeeMaths as lm

VeffEq = lm.exp("-(x**2+y**2)/2 - (1-u)/(((x+u)**2+y**2)**(1/2))-u/(((x+u-1)**2+y**2)**(1/2))")

mass_earth = 5.972 * 10 ** 24
mass_sun = 1.989 * 10 ** 30
mass_moon = 7.34767309 * 10 ** 22
mass_jupiter = 1.898 * 10 ** 27
mass_saturn = 5.683 * 10 ** 26

m1 = mass_jupiter
m2 = mass_saturn

u = 0.4  # m2 / (m1 + m2)

ThreeD = False
mode = 2
zoom = 1#1.2
res = 0.01 * zoom**2  # contour resolution
N = 1000  # number of contour lines

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
    return ((1 - u) * (x + u)) / ((x + u) ** 2 + y ** 2) ** (3 / 2) + (u * (x + u - 1)) / (
            (x + u - 1) ** 2 + y ** 2) ** (3 / 2) - x


def VeffdyHard(x, y, u):
    return ((1 - u) * y) / (y ** 2 + (x + u) ** 2) ** (3 / 2) + (u * y) / (y ** 2 + (x + u - 1) ** 2) ** (3 / 2) - y


def VeffHard(x, y, u):
    a = -(x ** 2 + y ** 2) / 2
    b = - (1 - u) / (np.sqrt((x + u) ** 2 + y ** 2))
    c = - u / (np.sqrt((x + u - 1) ** 2 + y ** 2))
    return a + b + c


def VeffdxdyHard(x, y, u):
    return (-3 * u * (-1 + u + x) * y) / ((-1 + u + x) ** 2 + y ** 2) ** (5 / 2) - (3 * (1 - u) * (u + x) * y) / (
            (u + x) ** 2 + y ** 2) ** (5 / 2)


def VeffdxdxHard(x, y, u):
    return -1 - (3 * u * (-1 + u + x) ** 2) / ((-1 + u + x) ** 2 + y ** 2) ** (5 / 2) + u / (
            (-1 + u + x) ** 2 + y ** 2) ** (3 / 2) - (3 * (1 - u) * (u + x) ** 2) / ((u + x) ** 2 + y ** 2) ** (
                   5 / 2) + (1 - u) / ((u + x) ** 2 + y ** 2) ** (3 / 2)


def VeffdydyHard(x, y, u):
    return -1 - (3 * u * y ** 2) / ((-1 + u + x) ** 2 + y ** 2) ** (5 / 2) + u / ((-1 + u + x) ** 2 + y ** 2) ** (
            3 / 2) - (3 * (1 - u) * y ** 2) / ((u + x) ** 2 + y ** 2) ** (5 / 2) + (1 - u) / (
                   (u + x) ** 2 + y ** 2) ** (3 / 2)


def D(x, y, u):
    return VeffdxdxHard(x, y, u) * VeffdydyHard(x, y, u) - (VeffdxdyHard(x, y, u)) ** 2


def f(state, t):
    x, y, v_x, v_y = state  # unpack state vector
    global u

    xdot = v_x
    ydot = v_y

    vxdot = - VeffdxHard(x, y, u) + (2 * v_y)
    vydot = - VeffdyHard(x, y, u) - (2 * v_x)
    return xdot, ydot, vxdot, vydot


t = np.arange(0.0, 27, 0.001)  # time steps
# fig, ax = plt.subplots()
fig = plt.figure()
if ThreeD:
    ax = fig.add_subplot(projection='3d')
else:
    ax = fig.add_subplot()

if not ThreeD:
    ax.grid()
#     ax.axvline(0, color='black')
#     ax.axhline(0, color='black')

# initial1 = [0.265, 0.867, 0, 0]
# states1 = odeint(f, initial1, t)
# flow1 = ax.plot(states1[:, 0], states1[:, 1], '', color='orange', zorder=100)[0]

# for x in np.arange(-1,1,0.4):
#     for y in np.arange(-1,1,0.4):
#         initial1 = [x, y, 0, 0]
#         states1 = odeint(f, initial1, t)
#         xx=states1[:, 0]
#         yy=states1[:, 1]
#         if(np.sqrt(xx[-1]**2+yy[-1]**2)<1):
#             flow1 = ax.plot(xx, yy)[0]

ax.plot(-u, 0, 'ro', zorder=100)
ax.plot(1 - u, 0, 'ro', zorder=100)

plt.title("u=" + str(u))
# + ", pos=(" + str(initial1[0]) + ", " + str(initial1[1]) + ")"
# + ", vel=(" + str(initial1[2]) + ", " + str(initial1[3]) + ")")

data = []
xd = []
yd = []

plt.xlim(-2*zoom, 2*zoom)
plt.ylim(-1.5*zoom, 1.5*zoom)

r = [[plt.xlim()[0] * zoom, plt.ylim()[0] * zoom], [plt.xlim()[1] * zoom, plt.ylim()[1] * zoom]]

ix = 0
for x in np.arange(r[0][0], r[1][0] + 0.0001, res):
    iy = 0
    data.append([])
    xd.append([])
    yd.append([])
    for y in np.arange(r[0][1], r[1][1] + 0.0001, res):
        xd[ix].append(x)
        yd[ix].append(y)

        # if mode == 1:
        #     dx = np.abs(VeffdxHard(x, y, u))
        #     dy = np.abs(VeffdyHard(x, y, u))
        #
        #     a = 0
        #     if dx < 0.04 and dy < 0.04:
        #         a = 5
        #     elif dx < 0.08 and dy < 0.08:
        #         a = 10
        #     elif dx < 0.12 and dy < 0.12:
        #         a = 15
        #     elif dx < 0.16 and dy < 0.16:
        #         a = 20
        #     data[ix].append(a)
        # elif mode == 2:
        #     a = D(x, y, u)  # * 10
        #     if not ThreeD:
        #         a = 10 ** a
        #         if a < -100:
        #             a = -100
        #         elif a > 100:
        #             a = 100
        #     data[ix].append(a)

        a = VeffHard(x, y, u)

        if a > 5:
            a = 5
        elif a < -5:
            a = -5

        # if a < -5100:
        #     print(a)
        #     a = -5100

        data[ix].append(a)
        iy += 1
    ix += 1

if not ThreeD:
    ax.contourf(xd, yd, data, N, extend='both')  # , colors=['#808080', '#A0A0A0', '#C0C0C0'])

if ThreeD:
    ax.plot_surface(xd, yd, np.array(data), zorder=-100)

plt.show()
