class subShell:
    def __new__(self, shellIndex, subShellIndex):
        return super().__new__(self)

    def __init__(self, shellIndex, subShellIndex):
        super().__init__()
        self.shellIndex = shellIndex
        self.subShellIndex = subShellIndex
        self.count = 0

    def setCount(self, amount):
        self.count = amount
        return self

    def subShellCharacter(self):
        if self.subShellIndex == 0:
            return 's'
        elif self.subShellIndex == 1:
            return 'p'
        elif self.subShellIndex == 2:
            return 'd'
        elif self.subShellIndex == 3:
            return 'f'
        elif self.subShellIndex == 4:
            return 'g'

    def electronCount(self):
        return self.count

    def getSubShellCapcity(self):
        if self.subShellIndex == 0:
            return 2
        elif self.subShellIndex == 1:
            return 6
        elif self.subShellIndex == 2:
            return 10
        elif self.subShellIndex == 3:
            return 14
        elif self.subShellIndex == 4:
            return 18

    def __str__(self):
        return str(self.shellIndex) + self.subShellCharacter() + toSuperScript(self.electronCount())

    def isFilled(self):
        return self.getSubShellCapcity() <= self.electronCount() or (self.shellIndex>=5 and self.subShellIndex == 2 and self.count == 1)

    def getNext(self, array):
        if self.shellIndex>=5 and self.subShellIndex == 2 and self.count == 1:
            return subShell(self.shellIndex-1, 3)

        if self.shellIndex == 1:
            return subShell(2, 0)
        elif self.shellIndex == 2 or self.shellIndex == 3:
            if self.subShellIndex == 0:
                return subShell(self.shellIndex, 1)
            elif self.subShellIndex == 2:
                return  subShell(self.shellIndex+1, 1)
            else:
                return  subShell(self.shellIndex+1, 0)
        elif self.shellIndex >= 4:
            if self.subShellIndex == 0:
                return subShell(self.shellIndex-1, 2)
            elif self.subShellIndex == 2:
                return  subShell(self.shellIndex+1, 1)
            elif self.subShellIndex == 3:
                return array[len(array)-2]
            else:
                return  subShell(self.shellIndex+1, 0)

        return subShell(0, 2).setCount(999)




class element:
    def __new__(self, atomicNumber):
        return super().__new__(self)

    def __init__(self, atomicNumber):
        super().__init__()
        self.atomicNumber = atomicNumber

    def getElectronConfig(self):
        configA = []
        number = self.atomicNumber

        current = subShell(1, 0)

        while number > 0:
            number -= 1
            current.setCount(current.count + 1)

            if current.isFilled():
                if not configA.__contains__(current):
                    configA.append(current)
                current = current.getNext(configA)


        if not current.isFilled() and current.count > 0:
            if not configA.__contains__(current):
                configA.append(current)

        return electronConfig(configA)

class electronConfig:
    def __new__(self, config):
        return super().__new__(self)

    def __init__(self, config):
        super().__init__()
        self.config = config  # []

    def __str__(self):
        out = ""
        for subShell in self.config:
            out += subShell.__str__()
        return out

def toSuperScriptSingle(num):
    d = {'0':"\u2070",
         '1':"\u00B9",
         '2':"\u00B2",
         '3':"\u00B3",
         '4':"\u2074",
         '5':"\u2075",
         '6':"\u2076",
         '7':"\u2077",
         '8':"\u2078",
         '9':"\u2079"}

    return d[num]

def toSuperScript(num):
    out = ""
    for char in str(num):
        out += toSuperScriptSingle(char)
    return out


#print(subShell(1,2).getNext())


print(element(75).getElectronConfig())
