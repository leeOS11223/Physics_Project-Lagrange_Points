import libs.LeeLibs as ll
import libs.LeeMaths as lm
np = ll.np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functools as ft

plt.rcParams['animation.ffmpeg_path'] = 'E:\Programs\\ffmpeg\\bin\\ffmpeg.exe'

class simulator:
    def __new__(cls):
        return super().__new__(cls)

    def __init__(self):
        super().__init__()
        self.objs = []
        self.tickCount = 0
        self.setCondition(tickLimitCondition(50))
        self.finished = False
        self.dimensions = 2
        self.timeRes = 1
        self.frames = 0
        self.frameData = []
        self.outputRegion = [[-5,5],[-5,5],[-5,5]]

    def add(self, obj):
        obj.setSimulator(self)
        obj.setup()
        self.objs.append(obj)

    def setCondition(self, con):
        con.setSim(self)
        self.condition = con
    
    def run(self):
        last = -1
        lastdata = []
        self.finished = False
        while not self.finished:
            frameData = []
            for obj in self.objs:
                if not obj.isDestroyed:
                    obj.tick(self.timeRes)

                if not obj.doNotPlot:
                    if not last == int(self.tickCount):
                        d = obj.getTickData()
                        frameData.append(d)

            if not last == int(self.tickCount):
                self.frameData.append(frameData)
                for obj in self.objs:
                    obj.addedFrameData(self.timeRes)

            for obj in self.objs:
                obj.actionTick(self.timeRes)
                obj.frame = self.frames

            if not last == int(self.tickCount):
                last = int(self.tickCount)
                self.frames += 1
                #print(self.frames)

            self.tickCount += 1 * self.timeRes


            if self.condition.check():
                self.finished = True

    def setTimeResolution(self, timeRes):
        self.timeRes = timeRes

    def plotResults(self, ftype = 'gif', frameOverride = -1, ThreeDOverride = False, extraplot = None):

        fig = plt.figure()
        if self.dimensions > 2 or ThreeDOverride:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()



        if not frameOverride == -1:
            self.animate(ax, extraplot, frameOverride)
            fig.show()
        else:
            ani = animation.FuncAnimation(fig, ft.partial(self.animate, ax, extraplot), blit=False, frames=self.frames)
            ani.save("output."+ftype)

    def animate(self, ax, extraplot, i):
        ax.clear()

        datax = []
        datay = []
        dataz = []
        for objData in self.frameData[i]:
            pos = objData[0]
            datax.append(pos[0])
            datay.append(pos[1])
            if self.dimensions > 2:
                dataz.append(pos[2])

        if not extraplot is None:
            extraplot(ax, i)

        if self.dimensions > 2:
            ax.plot(datax, datay, dataz, '.', color='black', marker='o', zorder=10)
        else:
            ax.plot(datax, datay, '.', color='black', marker='o', zorder=10)



        ax.set_xlim(self.outputRegion[0])
        ax.set_ylim(self.outputRegion[1])
        #if self.dimensions > 2:
        ax.set_zlim(self.outputRegion[2])

        return ax,

class simulatable:
    def __new__(cls):
        return super().__new__(cls)

    def __init__(self):
        super().__init__()
        self.simulationSpace = None
        self.isDestroyed = False
        self.doNotPlot = False
        self.frame = 0

    def setSimulator(self, sim: simulator):
        self.simulationSpace = sim

    def tick(self, timeRes):
        pass

    def addedFrameData(self, timeRes):
        pass

    def actionTick(self, timeRes):
        pass

    def setup(self):
        pass

    def getTickData(self):
        return None

class condition:
    def __new__(cls):
        return super().__new__(cls)

    def __init__(self):
        super().__init__()
        self.sim = None

    def setSim(self, sim):
        self.sim = sim

    def check(self):
        return False

class tickLimitCondition(condition):
    def __new__(cls, tickLimit):
        return super().__new__(cls)

    def __init__(self, tickLimit):
        super().__init__()
        self.tickLimit = tickLimit

    def check(self):
        return self.sim.tickCount >= self.tickLimit

class body(simulatable):
    def __new__(cls, mass):
        return super().__new__(cls)

    def __init__(self, mass):
        super().__init__()
        self.mass = mass
        self.position = []
        self.velocity = []
        self.nextposition = []
        self.nextvelocity = []
        self.acceleration = []
        self.static = False
        self.ignore = False

    def setup(self):
        d = self.getDimensions()
        for i in range(d):
            if len(self.position) < d:
                self.position.append(0)
                self.nextposition.append(0)

            if len(self.velocity) < d:
                self.velocity.append(0)
                self.nextvelocity.append(0)

            if len(self.acceleration) < d:
                self.acceleration.append(0)

    def setPosition(self, position):
        self.position = position

    def setMass(self, mass):
        self.mass = mass

    def setVelocity(self, velocity):
        self.velocity = velocity

    def setAcceleration(self, acceleration):
        self.acceleration = acceleration

    def addAcceleration(self, acceleration):
        for i in range(self.getDimensions()):
            self.acceleration[i] = acceleration[i]

    def getDimensions(self):
        return self.simulationSpace.dimensions

    def getOffset(self, other):
        o = []
        for i in range(self.getDimensions()):
            o.append(self.position[i] - other.position[i])
        return o

    # 2D only
    def getAngularOffset2D(self, other):
        off = self.getOffset(other)
        return np.arctan(off[1]/off[0])

    def getDistanceSquared(self, other):
        off = self.getOffset(other)
        o = 0
        for i in range(self.getDimensions()):
            o += off[i] * off[i]
        return o

    def getDistance(self, other):
        return np.sqrt(self.getDistanceSquared(other))

    def tickValues(self, timeRes):
        self.nextposition = self.position
        self.nextvelocity = self.velocity
        for i in range(self.getDimensions()):
            self.nextvelocity[i] += self.acceleration[i] * timeRes
            self.nextposition[i] += self.velocity[i] * timeRes

    def tick(self, timeRes):
        if self.static: return  # don't do anything if static
        self.tickValues(timeRes)

    def actionTick(self, timeRes):
        self.position = self.nextposition
        self.velocity = self.nextvelocity

    def getTickData(self):
        return [self.position.copy()]
