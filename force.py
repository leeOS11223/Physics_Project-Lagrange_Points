import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, odeint
import libs.LeeMaths as lm
import libs.LeeLibs as ll
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

VeffEq = lm.exp("-(x**2+y**2)/2 - (1-u)/(((x+u)**2+y**2)**(1/2))-u/(((x+u-1)**2+y**2)**(1/2))")

mass_earth = 5.972 * 10 ** 24
mass_sun = 1.989 * 10 ** 30
mass_moon = 7.34767309 * 10 ** 22
mass_jupiter = 1.898 * 10 ** 27
mass_saturn = 5.683 * 10 ** 26

m1 = mass_jupiter
m2 = mass_saturn

u = 0.4  # m2 / (m1 + m2)
finderCutOff = 16
finder = True
frameV = [0, 0]
simulationTime = 30
thirdBodyFromLagrangePoints = False
normalThirdBodySimulation = False


ThreeD = False
mode = 1
zoom = 1.2
res = 0.01  # contour resolution
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
    global u # constant

    xdot = v_x
    ydot = v_y

    vxdot = - VeffdxHard(x, y, u) + (2 * v_y)
    vydot = - VeffdyHard(x, y, u) - (2 * v_x)
    return xdot, ydot, vxdot, vydot


t = np.arange(0.0, simulationTime, 0.001)  # time steps
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

if normalThirdBodySimulation:
    initial1 = [0.4, 0.9, 0, 0]
    states1 = odeint(f, initial1, t)
    flow1 = ax.plot(states1[:, 0], states1[:, 1], '', color='orange', zorder=100)[0]

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

if not frameV == [0, 0]:
    plt.title("u=" + str(u) + ", frameV=" + str(frameV))
else:
    if normalThirdBodySimulation:
        plt.title("u=" + str(u)
         + ", pos=(" + str(initial1[0]) + ", " + str(initial1[1]) + ")"
         + ", vel=(" + str(initial1[2]) + ", " + str(initial1[3]) + ")")
    else:
        plt.title("u=" + str(u))

data = []
xd = []
yd = []

plt.xlim(-2, 2)
plt.ylim(-1.5, 1.5)

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

        if mode == 1:
            dx = np.abs(-VeffdxHard(x, y, u) + 2 * frameV[1])
            dy = np.abs(-VeffdyHard(x, y, u) - 2 * frameV[0])

            a = 0
            if dx < 0.005 and dy < 0.005:
                a = 35
            elif dx < 0.01 and dy < 0.01:
                a = 30
            elif dx < 0.02 and dy < 0.02:
                a = 25
            elif dx < 0.04 and dy < 0.04:
                a = 20
            elif dx < 0.08 and dy < 0.08:
                a = 15
            elif dx < 0.12 and dy < 0.12:
                a = 10
            elif dx < 0.16 and dy < 0.16:
                a = 5
            data[ix].append(a)
        elif mode == 2:
            a = D(x, y, u)  # * 10
            if not ThreeD:
                a = 10 ** a
                if a < -100:
                    a = -100
                elif a > 100:
                    a = 100
            data[ix].append(a)

        # data[ix].append(VeffdxHard(x, y, u) * VeffdyHard(x, y, u))
        iy += 1
    ix += 1

b = np.array(data)


# shape = np.shape(b)
# b = np.where(b == np.max(b))
#
# peaks = np.empty(shape)
# for i in range(np.shape(b)[1]):
#     iX = b[0][i]
#     iY = b[1][i]
#     peaks[iX][iY] = 1
#     ax.plot(xd[iX][iY], yd[iX][iY], 'b.', zorder=100)


# def detect_peaks(arr):
#     peaks = []
#     for i in range(1, arr.shape[0] - 1):
#         for j in range(1, arr.shape[1] - 1):
#             if arr[i, j] > arr[i - 1, j] and arr[i, j] > arr[i + 1, j] and arr[i, j] > arr[i, j - 1] and arr[i, j] > \
#                     arr[i, j + 1]:
#                 peaks.append((i, j))
#     return peaks
#
# d = detect_peaks(np.array(data))
# print(d)
#
# for i in d:
#     ax.plot(xd[i[0]][i[1]], yd[i[0]][i[1]], 'b.', zorder=100)

# shape = np.shape(b)
# check = np.empty(shape)
# for x in range(shape[0]):
#     for y in range(shape[1]):
#         if check[x][y] == -1:  # skip if done
#             continue
#         check[x][y] = -1
#
#         d = data[x][y]
#         if not d == 0:
#             print(d)

def walk(data, point1, point2):
    stack = []
    doneStack = []
    stack.append(point1)

    while len(stack) > 0:
        current = stack.pop(0)
        doneStack.append(current)

        for xx in range(-2, 3):
            for yy in range(-2, 3):
                next = [current[0] + xx, current[1] + yy]
                if not data[next[0]][next[1]] < finderCutOff: #16
                    if not (stack.__contains__(next) or doneStack.__contains__(next)):
                        stack.append(next)
                if doneStack.__contains__(point2): return True

        # above = [current[0], current[1] + 1]
        # if not data[above[0]][above[1]] == 0:
        #     if not (stack.__contains__(above) or doneStack.__contains__(above)):
        #         stack.append(above)
        #
        # below = [current[0], current[1] - 1]
        # if not data[below[0]][below[1]] == 0:
        #     if not (stack.__contains__(below) or doneStack.__contains__(below)):
        #         stack.append(below)
        #
        # left = [current[0] - 1, current[1]]
        # if not data[left[0]][left[1]] == 0:
        #     if not (stack.__contains__(left) or doneStack.__contains__(left)):
        #         stack.append(left)
        #
        # right = [current[0] + 1, current[1]]
        # if not data[right[0]][right[1]] == 0:
        #     if not (stack.__contains__(right) or doneStack.__contains__(right)):
        #         stack.append(right)

    return doneStack.__contains__(point2)

if finder:
    xpeaks = []
    ypeaks = []
    for y in range(np.shape(data)[0]):
        # y=int(len(b)/2)
        peaks, _ = find_peaks(b[y], height=finderCutOff)
        for peak in peaks:
            xpeaks.append([y, peak])
            # ax.plot(xd[y][peak], yd[y][peak], 'b.', zorder=100)

    for y in range(np.shape(data)[1]):
        peaks, _ = find_peaks(b.transpose()[y], height=finderCutOff)
        for peak in peaks:
            ypeaks.append([peak, y])
            # ax.plot(xd[peak][y], yd[peak][y], 'y.', zorder=100)

    peaks = []

    for peak in xpeaks:
        if ypeaks.__contains__(peak):
            peaks.append(peak)

    print(len(peaks))

    newpeaks = peaks.copy()

    i = 0
    for peak1 in newpeaks:
        i += 1
        for peak2 in newpeaks:
            if peak2 is None: continue
            if peak1 is None: break
            if not peak1 == peak2:
                walked = walk(b, peak1, peak2)
                if walked:
                    peak1V = b[peak1[0], peak1[1]]
                    peak2V = b[peak2[0], peak2[1]]

                    if peak1V > peak2V:
                        newpeaks[newpeaks.index(peak2)] = None
                    else:
                        newpeaks[newpeaks.index(peak1)] = None
                        break

        print(str(int(i / (len(newpeaks)) * 100)) + "%")

    peaks = []
    for peak in newpeaks:
        if not peak is None:
            peaks.append(peak)

    lpoints = [["name","x","y"]]

    for peak in peaks:
        ax.plot(xd[peak[0]][peak[1]], yd[peak[0]][peak[1]], 'r.', zorder=100)

        posx,posy = xd[peak[0]][peak[1]], yd[peak[0]][peak[1]]

        # round to nearest 4 dp
        posx = round(posx * 10000) / 10000
        posy = round(posy * 10000) / 10000

        lpoints.append(["",posx,posy])

        print(posx,posy)

        if thirdBodyFromLagrangePoints:
            initial1 = [xd[peak[0]][peak[1]], yd[peak[0]][peak[1]], frameV[0], frameV[1]]
            states1 = odeint(f, initial1, t)
            flow1 = ax.plot(states1[:, 0], states1[:, 1], '', zorder=100)[0]

    # label each point  L1, L2, L3, L4, L5
    # L1 if its in the centre
    # L2 if its in the right
    # L3 if its in the left
    # L4 if its in the top
    # L5 if its in the bottom
    # Create a list of tuples containing the indices and coordinates of each point
    points = []
    for i, point in enumerate(lpoints[1:]):
        x, y = point[1], point[2]
        points.append((i, x, y))

    # Sort the points by x-coordinate
    points.sort(key=lambda p: p[1])

    # Label the points
    for i, point in enumerate(points):
        idx, x, y = point
        if i == 0:
            lpoints[idx+1][0] = "L3"
        elif i == len(points) - 1:
            lpoints[idx+1][0] = "L2"
        else:
            lpoints[idx+1][0] = ""

    points.sort(key=lambda p: p[2])

    # Label the points
    for i, point in enumerate(points):
        idx, x, y = point
        if i == 0:
            lpoints[idx+1][0] = "L5"
        elif i == len(points) - 1:
            lpoints[idx+1][0] = "L4"
        else:
            if lpoints[idx+1][0] == "":
                lpoints[idx+1][0] = "L1"

    lpoints = ll.toDataObject(lpoints)

    print(lpoints.asLatexTable("A table showing the positions of the the lagrangian points"))
    print()
    print(lpoints.toArray())

if not ThreeD:
    ax.contourf(xd, yd, data, N, extend='both')  # , colors=['#808080', '#A0A0A0', '#C0C0C0'])

if ThreeD:
    ax.plot_surface(xd, yd, np.array(data), zorder=-100)

plt.show()
