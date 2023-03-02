def Veffdx(x, y, u):
    return 0
def Veffdy(x, y, u):
    return 0

def f(state, t):
    x, y, v_x, v_y = state  # unpack state vector
    global u # constant

    xdot = v_x
    ydot = v_y

    vxdot = - Veffdx(x, y, u) + (2 * v_y)
    vydot = - Veffdy(x, y, u) - (2 * v_x)
    return xdot, ydot, vxdot, vydot