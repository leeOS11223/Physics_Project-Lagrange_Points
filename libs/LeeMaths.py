import libs.LeeLibs as ll
sym = ll.sym

version = 2

class expression:
    expression = None

    def __new__(self, expressionAsString="", evaluate=False, expression=None, internalLeeSyntax=True):
        return super().__new__(self)

    def __init__(self, expressionAsString="", evaluate=False, expression=None, internalLeeSyntax=True):
        expressionAsString=str(expressionAsString)
        if internalLeeSyntax:
            expressionAsString = addLeeSyntax(expressionAsString)

        if expressionAsString.__contains__("="):
            parts = expressionAsString.split("=")
            self.expression = (exp(parts[1]) - exp(parts[0])).expression
            return super().__init__()

        if len(expressionAsString) > 0:
            self.expression = sym.parse_expr(expressionAsString, evaluate=evaluate)
        if expression is not None:
            self.expression = expression
        return super().__init__()

    def getValues(self, funcSymbol, varSymbol, data=None):
        if isinstance(varSymbol, dict):
            dicts = []
            for i in range(len(list(varSymbol.values())[0])):
                dicts.append(dict())
            for key in varSymbol.keys():
                i=0
                for value in varSymbol[key]:
                    dicts[i][key]=value
                    i+=1
            OData = []
            func = self.solve(funcSymbol)
            for dd in dicts:
                OData.append(func[0].evaluate(dd))

            return OData
        else:
            values = []
            for dd in data:
                func = self.solve(funcSymbol)
                d = dict()
                d[varSymbol]=dd
                values.append(func[0].evaluate(d))
            return values


    def substitute(self, symbol, expression2=None, evaluate=True):
        symbol = "var_" + symbol
        if isinstance(symbol, dict):
            e = self
            for ex in symbol.keys():
                e = e.substitute(str(ex),symbol[ex], evaluate=evaluate)
            return e

        if not isinstance(expression2, exp):
            expression2 = exp(expression2)

        return expression(expression=self.expression.subs(symbol, expression2.expression, evaluate=evaluate))

    def evaluate(self, subs):
        newsubs = dict()
        for sub in subs:
            newsubs["var_"+sub] = subs[sub]
        subs=newsubs
        try:
            hasError = False
            totalError = 0
            for sub in subs:
                if isinstance(subs[sub], ll.errorFloat):
                    hasError = True
                    partial = sym.diff(self.expression, sub).evalf(subs=subs)
                    error = pow(partial*subs[sub].toError(), 2).__float__()
                    totalError += error

            if hasError:
                return ll.toErrorFloatSingle(self.expression.evalf(subs=subs).__float__(), ll.np.sqrt(totalError))
            else:
                return self.expression.evalf(subs=subs).__float__()
        except:
            print(f"\033[91mError: Make sure to sub in values for everything!\nFound variables: {self.expression.free_symbols}\033[0m")
            return

    def __add__(self, other):
        ex = None
        if isinstance(other, exp):
            ex = other
        else:  # number?
            ex = exp(str(other.__float__()))

        ex2 = exp("internal_a+internal_b")
        ex2 = ex2.substitute("internal_a", self)
        ex2 = ex2.substitute("internal_b", ex)
        return ex2

    def __sub__(self, other):
        ex = None
        if isinstance(other, exp):
            ex = other
        else:  # number?
            ex = exp(str(other.__float__()))

        ex2 = exp("internal_a-internal_b")
        ex2 = ex2.substitute("internal_a", self)
        ex2 = ex2.substitute("internal_b", ex)
        return ex2

    def __mul__(self, other):
        ex = None
        if isinstance(other, exp):
            ex = other
        else:  # number?
            ex = exp(str(other.__float__()))

        ex2 = exp("internal_a*internal_b")
        ex2 = ex2.substitute("internal_a", self)
        ex2 = ex2.substitute("internal_b", ex)
        return ex2

    def __truediv__(self, other):
        ex = None
        if isinstance(other, exp):
            ex = other
        else:  # number?
            ex = exp(str(other.__float__()))

        ex2 = exp("internal_a/internal_b")
        ex2 = ex2.substitute("internal_a", self)
        ex2 = ex2.substitute("internal_b", ex)
        return ex2

    def __pow__(self, other):
        ex = None
        if isinstance(other, exp):
            ex = other
        else:  # number?
            ex = exp(str(other.__float__()))

        ex2 = exp("internal_a**internal_b")
        ex2 = ex2.substitute("internal_a", self)
        ex2 = ex2.substitute("internal_b", ex)
        return ex2

    def __abs__(self):
        ex2 = exp(expression=sym.Abs(self.expression))
        return ex2

    def solve(self, symbol):
        symbol="var_"+symbol
        outE = []
        solved = sym.solve(self.expression, symbol)
        for s in solved:
            outE.append(exp(expression=s))
        return outE

    def encapsulate(self):
        ex2 = exp("(a)")
        ex2 = ex2.substitute("a", self, evaluate=False)
        return ex2

    def differentiate(self, withRespect=None, evaluate=True):
        if withRespect is None:
            withRespect = exp("x")
        #withRespect=withRespect
        return exp(expression=sym.diff(self.expression, withRespect.expression, evaluate=evaluate))
    diff = differentiate

    def integrate(self, withRespect=None):
        if withRespect is None:
            withRespect = exp("x")
        withRespect="var_"+withRespect
        return exp(expression=sym.integrate(self.expression, withRespect.expression))
    diff = differentiate

    def __str__(self):
        return removeLeeSyntax(self.pretty())

    def pretty(self):
        return removeLeeSyntax(sym.pretty(self.expression))

    def ugly(self):
        return removeLeeSyntax(self.expression.__str__(), True)

    def __float__(self):
        return expression.__float__()

    def __repr__(self):
        return removeLeeSyntax(self.ugly())#"expression("+")"

    def getSymbols(self):
        symbols = []
        for s in self.expression.free_symbols:
            symbols.append(exp(s.__str__()))
        return symbols

    def toLatex(self):
        return removeLeeSyntax(sym.latex(self.expression), True)
exp = expression

class function():
    funVar = "x"
    exp

    def __new__(self, variable, expression):
        return super().__new__(self)

    def __init__(self, variable, expression):
        self.funVar = variable
        self.exp = expression
        return super().__init__()

    def substitute(self, expression):
        return self.toExpression().substitute(self.exp, self.funVar, expression, True)

    def toExpression(self):
        return exp

    def __str__(self):
        return self.funVar + " = " + self.exp.__str__()

    def __repr__(self):
        return self.funVar + " = " + self.exp.__repr__()

fun = function

def addLeeSyntax(text):
    current = False
    newtext = ""
    for c in text:
        cCode = ord(c[0].lower())
        if cCode>=97 and cCode<=122 or c[0].lower() == "_": # is letter
            if not current:
                current = True
                newtext += "var_"
        else:
            current = False
        newtext+=c
    return newtext

def removeLeeSyntax(text, noSpaces=False):#, noSpaces=False):
    if noSpaces:
        return str(text).replace("var_", "")
    return str(text).replace("var_", "    ")

class vector:
    array = []
    symbols = []

    def __new__(self, expres, coords):
        return super().__new__(self)

    def __init__(self, expres, coords):
        self.array = []
        self.symbols = coords.split(",")
        for s in self.symbols:
            p=expres.diff(exp(s))
            self.array.append(p)

        return super().__init__()

    def toExpression(self):
        ex = self.array[0].encapsulate() * exp(self.symbols[0])
        for i in range(1, len(self.symbols)):
            ex += self.array[i].encapsulate() * exp(self.symbols[i])
        return ex

    def __str__(self):
        return self.toExpression().pretty()

    def __mul__(self, other):
        e = other
        if isinstance(other, vector):
            other.toExpression()
        ex = self.toExpression() * e
        symbols = ""


        if isinstance(other, vector):
            us = []
            for s in self.symbols + other.symbols:
                if not us.__contains__(s):
                    us.append(s)
                    symbols += s + ","
        else:
            us = []
            for s in self.symbols:
                if not us.__contains__(s):
                    us.append(s)
                    symbols += s + ","

        return vector(ex, symbols[:-1])

    def __add__(self, other):
        ex = other.toExpression()+self.toExpression()
        symbols = ""

        us = []
        for s in self.symbols+other.symbols:
            if not us.__contains__(s):
                us.append(s)
                symbols += s+","
        return vector(ex, symbols[:-1])

    def __sub__(self, other):
        ex = self.toExpression()-other.toExpression()
        symbols = ""

        us = []
        for s in self.symbols+other.symbols:
            if not us.__contains__(s):
                us.append(s)
                symbols += s+","
        return vector(ex, symbols[:-1])

    def toLatex(self):
        array = []
        for a in self.array:
            array.append(a.expression)

        return sym.latex(sym.Matrix(ll.rotateArray([array])))





