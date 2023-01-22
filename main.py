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
        print(a)
        self.nextAcceleration = a

    def actionTick(self, timeRes):
        if self.isDestroyed: return
        self.setAcceleration(self.nextAcceleration)

class lagrangeSampler(obj):
    def __new__(cls, sampleRange):
        return super().__new__(cls, 0)

    def __init__(self, sampleRange):
        super().__init__(0)
        self.static = True
        self.doNotPlot = True
        self.isDestroyed = True
        self.sampleRange = sampleRange

    def actionTick(self, timeRes):
        pass

r = Sun_Earth_distance*1.2

simSpace = sim.simulator()
simSpace.setCondition(sim.tickLimitCondition(32))
simSpace.setTimeResolution(0.1)
#simSpace.outputRegion= [[-r,r],[-r,r],[-r,r]]

simSpace.dimensions = 3

o = obj(10)
o.static = True
simSpace.add(o)

o2 = obj(1)
o2.setPosition([0, 3, 0])
o2.setVelocity([-0.740159, -0.2245, 0])
simSpace.add(o2)


s = lagrangeSampler([[-50,50],[-50,50],[-50,50]])
#simSpace.add(s)


simSpace.run()
simSpace.plotResults('gif')







