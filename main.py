import simulator as sim

class obj(sim.simulatable):
    def __new__(cls):
        return super().__new__(cls)

    def __init__(self):
        super().__init__()
        self.i = 0

    def tick(self):
        self.i+=1
        print("test "+str(self.i))

simSpace = sim.simulator()
simSpace.setCondition(sim.tickLimitCondition(5))

o = obj()
simSpace.add(o)

simSpace.run()









