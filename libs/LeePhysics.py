import LeeLibs as ll
import LeeMaths as lm


class Unit:
    def __new__(self, value):
        return super().__new__(self)

    def __init__(self, value):
        super().__init__()
        if type(value) is lm.exp:
            self.value = value
        else:
            self.value = lm.exp(value)

    def __str__(self):
        return self.value.ugly()

    def __mul__(self, other):
        return Unit(self.value * other)

    def toSI(self):
        return self


class CompoundUnit(Unit):
    def __new__(self, symbol, expression, dictionary):
        return super().__new__(self, symbol)

    def __init__(self, symbol, expression, dictionary):
        super().__init__(symbol)
        if type(expression) is lm.exp:
            self.exp = expression
        else:
            self.exp = lm.exp(expression)

        for key in dictionary:
            self.exp = self.exp.substitute(key, dictionary[key])

    def __str__(self):
        return super().__str__()

    def toSI(self):
        return Unit(self.exp)


class unitFloat:
    def __new__(self, value, unit):
        return super().__new__(self)

    def __init__(self, value, unit):
        super().__init__()
        self.value = value
        if type(unit) is Unit:
            self.unit = unit
        else:
            self.unit = Unit(unit)

    def __str__(self):
        return str(self.value) + " [" + str(self.unit) + "]"

    def __float__(self):
        return self.value.__float__()
