import simulator as sim
import libs.LeeMaths as lm
import libs.LeeLibs as ll
np = ll.np

Newtons_law_of_universal_gravitation = lm.exp("G*m1*m2/(r**2)")
Gravitational_constant = 1#lm.exp("6.67430 * 10**-11").evaluate(dict())  # N m^2 kg^-1

Sun_Earth_distance = 147.22 * 10**6 #km
mass_sun = 1.989 * 10**30 #kg
mass_earth = 5.972 * 10**24 #kg
mass_ratio = mass_sun/mass_earth

class obj(sim.body):
    def __new__(cls, mass):
        return super().__new__(cls, mass)

    def __init__(self, mass):
        super().__init__(mass)
        self.nextAcceleration = [0, 0, 0]

    def getOtherObjects(self):
        sbs = self.simulationSpace.objs.copy()

        if self in sbs:
            sbs.remove(self)

        for o in sbs:
            if o.isDestroyed:
                sbs.remove(o)

        return sbs

    def calculateForceOfGravityBetweenTwoBodies(self, other):
        offset = self.getOffset(other)
        force = Newtons_law_of_universal_gravitation.evaluate(
            dict(G=Gravitational_constant, m1=self.mass, m2=other.mass, r=self.getDistanceSquared(other)))

        return list(-1 * np.array(ll.normalise(offset)) * force)

    def forceToAcceleration(self, force):
        self.o = []
        for i in range(self.getDimensions()):
            self.o.append(force[i] / self.mass)
        return self.o

    def tick(self, timeRes):
        super(obj, self).tick(timeRes)

        a = []
        for other in self.getOtherObjects():
            t=self.forceToAcceleration(self.calculateForceOfGravityBetweenTwoBodies(other))
            for i in range(self.getDimensions()):
                if len(a) <= i:
                    a.append(0)
                a[i] += t[i]
        #print(a)
        self.nextAcceleration = a

    def actionTick(self, timeRes):
        if self.isDestroyed: return
        self.setAcceleration(self.nextAcceleration)

class gravitySampler(obj):
    def __new__(cls, simSpace, sampleRange):
        return super().__new__(cls, 0)

    def __init__(self, simSpace, sampleRange):
        super().__init__(1)
        self.static = True
        self.doNotPlot = True
        self.isDestroyed = True
        self.sampleRange = sampleRange
        self.setSimulator(simSpace)
        self.last = -1

    def getGravityVector(self, pos):
        self.setPosition(pos)
        a = []
        for other in self.getOtherObjects():
            t = self.forceToAcceleration(self.calculateForceOfGravityBetweenTwoBodies(other))
            for i in range(self.getDimensions()):
                if len(a) <= i:
                    a.append(0)
                a[i] += t[i]
        return a

    def getGravityStrength(self, pos):
        strength = self.getGravityVector(pos)

        i = 0
        for d in range(self.getDimensions()):
            i += strength[d]**2

        if np.abs(i) > 5:
            return 10
        return np.sqrt(i)

    def getField(self, sampleSize = 1):
        outx = np.linspace(self.sampleRange[0][0],self.sampleRange[0][1], sampleSize)
        outy = np.linspace(self.sampleRange[1][0],self.sampleRange[1][1], sampleSize)
        outz = []

        global star

        for x in outx:
            az = []
            for y in outy:
                strength = self.getGravityStrength([x, y])

                spinning = - self.getDistance(star) * 0.4

                az.append(strength - spinning)
            outz.append(az)

        return [np.array(outx), np.array(outy), np.array(outz)]

    def addedFrameData(self, timeRes):
        global fielddata
        fielddata.append(s.getField(20))


fielddata = []
star = None

def extraplot(ax, i):
    x, y = np.meshgrid(fielddata[i][0], fielddata[i][1])
    #print(fielddata[0])
    #mycmap = ll.plt.get_cmap('gist_earth')
    ax.plot_surface(y, x,-fielddata[i][2], zorder=-100)#, cmap=mycmap) #plot_wireframe


r = Sun_Earth_distance*1.2

simSpace = sim.simulator()
simSpace.setCondition(sim.tickLimitCondition(0))
simSpace.setTimeResolution(1)
#simSpace.outputRegion= [[-r,r],[-r,r],[-r,r]]

simSpace.dimensions = 2

star = obj(10)
star.setPosition([0, 1, 0])
star.static = True
simSpace.add(star)

o2 = obj(1)
o2.setPosition([0, -2, 0])
o2.setVelocity([-0.740159, -0.2245, 0])
simSpace.add(o2)

r = 5
s = gravitySampler(simSpace, [[-r,r],[-r,r],[-r,r]])
simSpace.add(s)
#fielddata = s.getField(15)
#print(fielddata)



simSpace.run()
#simSpace.plotResults(frameOverride=0, ThreeDOverride = True, extraplot=extraplot)
simSpace.plotResults(ThreeDOverride = True, extraplot=extraplot)







