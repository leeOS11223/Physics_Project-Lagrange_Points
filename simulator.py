class simulatable:
    def __new__(cls):
        return super().__new__(cls)

    def __init__(self):
        super().__init__()
        
    def tick(self):
        pass

class simulator:
    def __new__(cls):
        return super().__new__(cls)

    def __init__(self):
        super().__init__()
        self.objs = []
        self.tickCount = 0
        self.setCondition(tickLimitCondition(50))
        self.finished = False

    def add(self, obj):
        self.objs.append(obj)

    def setCondition(self, con):
        con.setSim(self)
        self.condition = con
    
    def run(self):
        self.finished = False
        while not self.finished:
            for obj in self.objs:
                obj.tick()
            self.tickCount+=1
            if self.condition.check():
                self.finished = True


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



