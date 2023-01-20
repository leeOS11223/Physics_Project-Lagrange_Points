import array
import math

import numpy as np
import datetime as dt
import re
import latextable
import sympy as sym
from texttable import Texttable
import matplotlib.pyplot as plt

version = 2

#convert string with engineering units to float
def convert(x):
    if "ms" in str(x):
        return float(re.sub("[^0-9.]", "", str(x).replace("ms", "")))/1000
    elif "us" in str(x):
        return float(re.sub("[^0-9.]", "", str(x).replace("us", "")))/1000000
    elif "ns" in str(x):
        return float(re.sub("[^0-9.]", "", str(x).replace("ns", "")))/1000000000
    else:
        return float(x)

# parse a DateTime from a string
def parseDateTime(text, timeFormat="%Y-%m-%d %H:%M:%S"):
    try:
        return dt.datetime.strptime(text, timeFormat)
    except ValueError as err:
        return text

#https://stackoverflow.com/a/2566508
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# get date only
def dateOnly(dateTime):
    if dateTime == None:
        return dateTime
    try:
        return dateTime.date()
    except ValueError as err:
        return dateTime


# get time only
def timeOnly(dateTime):
    if dateTime == None:
        return dateTime
    try:
        return dateTime.time()
    except ValueError as err:
        return dateTime


# parse a float from a string
def parseFloat(text):
    if text == None:
        return text
    try:
        return float(text)
    except ValueError as err:
        return text


# parse a bool from a string
def parseBool(text):
    if text == None:
        return text
    try:
        return bool(text)
    except ValueError as err:
        return text


# load a file as a 2D array
def convertFileToArray(path, rotate=False, npArray=True):
    outData = []
    f = open(path, "r")
    lines = f.read().split("\n")
    rawData = []
    for line in lines:
        rawData.append(line.split(","))
    if rotate:
        outData = rotateArray(rawData)
    else:
        outData = rawData
    if npArray:
        outData = np.array(outData)
    return outData


# run a single input method on every item in an array
def runMethodOnArray(array, method):
    i = 0
    for item in array:
        array[i] = method(item)
        i += 1
    return array


# replace every instance of oldData for newData in the entire 2D Data Structure.
def replaceData(array, oldData, newData):
    for y in array:
        i = 0
        for x in y:
            if x == oldData:
                y[i] = newData
            i += 1


# get the array which the first item matches text
def getArrayFromFirst(array, text):
    for i in range(0, len(array)):
        if str(array[i][0]).__contains__(text):
            return array[i][1:len(array[i])]
    return None


# flip the x and y cords of an array.
# i.e: array[x][y] -> array[y][x]
def rotateArray(rawData, flattenArray=False):
    data = []
    for part in rawData[0]:
        data.append([])
    for part in rawData:
        i = 0
        for part2 in part:
            data[i].append(part2)
            i += 1
    tempData = []
    if flattenArray:
        for d in data:
            for e in d:
                tempData.append(e)
        return tempData
    return data


# pops all the data points will null in them, if only one null is found in an array,
# i.e if the array [null ,1.2, 6.5] is found the whole array will be popped.
def removeNulls(array):
    i = 0
    while i < len(array):
        a = array[i]
        for d in a:
            if d is None:
                array.pop(i)
                i -= 1
                break
        i += 1
    return array

def removeAtindexs(array, indexs):
    i = 0
    for d in array:
        if not indexs[i]:
            array = np.delete(array,i)
            #array.remove(i)
            i -= 1
        i += 1
    return array


def amplitude(array):
    return np.max(array)-np.min(array)

def asLatexTable(array, firstRow, caption=None, label=None, precision=3, longTable=False):
    n = rotateArray(array)
    n.insert(0, firstRow)

    align = ["l"]#"r"]
    valign = ["t"]

    for i in range(0,len(firstRow)-1):
        align.append("p{3cm}")
        valign.append("m")

    table = Texttable()
    table.set_precision(precision)
    table.set_cols_align(align)
    table.set_cols_valign(valign)
    table.add_rows(n)

    if caption is None:
        caption="temp"
    if label is None:
        label="temp"
    if not longTable:
        return latextable.draw_latex(table, caption=caption, label="tab:"+label).replace("\\begin{table}", "\\begin{table}[H]")
    else:
        return latextable.draw_latex(table, caption=caption, label="tab:"+label).replace("\\begin{table}", "\\begin{longtable}").replace("\\end{table}", "\\end{longtable}")

def asTextTable(array, firstRow, precision=3):
    if array is None:
        n = [firstRow]
    else:
        n = rotateArray(array)
        n.insert(0, firstRow)

    align = ["l"]
    valign = ["t"]

    for i in range(0,len(firstRow)-1):
        align.append("r")
        valign.append("m")

    table = Texttable()
    table.set_precision(precision)
    table.set_cols_align(align)
    table.set_cols_valign(valign)
    table.add_rows(n)
    if array is None:
        d = table.draw()
        r = d.split("\n")
        s = ""

        for i in range(len(r)):
            if i == len(r)-2:
                continue
            s += r[i] + "\n"
        return s[:-1]
    else:
        return table.draw()

# https://stackoverflow.com/a/17452783 by atomh33ls
def argand(a, polar=False):
    import matplotlib.pyplot as plt
    if(polar):
        for x in a:
            plt.polar([0, np.angle(x)], [0, abs(x)], marker='o')
    else:
        for x in range(len(a)):
            plt.plot([0,a[x].real],[0,a[x].imag],'ro-')
        limit=np.max(np.ceil(np.absolute(a))) # set limits for axis
        plt.xlim((-limit,limit))
        plt.ylim((-limit,limit))
    plt.ylabel('Imaginary', labelpad=27)
    plt.xlabel('Real')
    return plt

def sampleDeviation(array):
    sD=0
    mean = np.mean(array)
    for i in array:
        sD+=np.power(i-mean, 2)
    sD*=1/(len(array)-1)

    sD=np.sqrt(sD)
    return sD

def calculateError(array):
    sD = sampleDeviation(array)
    return sD/np.sqrt(len(array))

class engineeringFloat(float):
    def __new__(self, value):
        return super().__new__(self, value)

   # def __init__(self):
   #     super().__init__()

    def toSignificantFigure(self, digits):
        return engineeringFloat(round(self.__float__(), digits - int(math.floor(math.log10(abs(self.__float__())))) - 1))

    def __float__(self):
        return super().__float__()

    def __str__(self):
        if self.__float__()==0:
            return 0
            #return super().__str__()

        tS=1
        if abs(self.__float__())<1:
            tS=-1

        #OVERALLSIGN = np.sign(self.__float__())
        amount = np.floor(np.log10(np.abs(self.__float__())))
        sign = -tS*np.sign(amount) #reversed this, may be wrong
        si = np.floor(amount/3)

        offset = (si*3)
        temp = self.__float__()
        if sign == -1:
            temp *= pow(0.1, offset)
        elif sign == 1:
            temp *= pow(10, offset)
        return str(round(temp,3).__repr__() + getEngenneringUnit(si))

    def __repr__(self):
        return str(super().__repr__())
        #return "engineeringFloat("+str(super().__repr__())+")"

class errorFloat(engineeringFloat):
    def __new__(self, value, error, toFractional=False):
        return super().__new__(self, value)

    def __init__(self, value, error, toFractional=False):
        super().__init__()
        if toFractional:
            self.error = engineeringFloat(error)#self.toErrorFractional(error)
        else:
            self.error = engineeringFloat(self.fromErrorFractional(error))

    def super(self):
        return super()

    def __str__(self):
        rp = self.reducePrecision()
        if self.toError() != 0:
            return str(rp.super().__str__())+u" \u00B1 " + str(rp.toError())
        return str(rp.super().__str__())

    def __repr__(self):
        rp = self.reducePrecision()
        if self.toError() != 0:
            return str(rp.super().__repr__())+u" \u00B1 " + str(rp.toError())
        return str(rp.super().__repr__())
        #return "errorFloat("+str(super().__repr__())+u" \u00B1 " + str(self.toError())+")"

    def reducePrecision(self):
        err = reducePrecision(self.toError())
        v = self.__float__()
        d = (10**err[1])
        if err[1] < 0:
            return toErrorFloatSingle(round(float(v), int(-err[1])), err[0])
        else:
            r = v/(d*10)
            return toErrorFloatSingle(round(r) * (d*10), err[0])

    def __add__(self, other):
        e=pow(self.toError(),2)
        if type(other) is type(self):
            e+=pow(other.toError(),2)
        return errorFloat(self.__float__()+other.__float__(), np.sqrt(e), True)

    def __abs__(self):
        return errorFloat(abs(self.toFloat()), abs(self.toError()), True)

    def __sub__(self, other):
        e=pow(self.toError(),2)
        if type(other) is type(self):
            e+=pow(other.toError(),2)
        return errorFloat(self.__float__()-other.__float__(), np.sqrt(e), True)

    def __mul__(self, other):
        if type(other) is type(self):
            e=pow(self.fractionError(),2)
            e+=pow(other.fractionError(),2)
            return errorFloat(self.__float__()*other.__float__(), np.sqrt(e))
        else:
            e = self.toError()
            e*=other
            return errorFloat(self.__float__() * other.__float__(), e, True)


    def __round__(self, amount):
        return errorFloat(round(self.toFloat(), amount),round(self.toError(), amount),True)

    def round(self, amount, errorAmount=-1):
        if errorAmount==-1:
            errorAmount=amount
        return errorFloat(round(self.toFloat(), amount), round(self.toError(), errorAmount),True)

    def sqrt(self):
        return errorFloat(np.sqrt(self.toFloat()),self.fractionError()/2)

    def inverseNegative(self):
        return errorFloat(-self.toFloat(),self.fractionError()/2)

    def reciprocal(self):
        return errorFloat(1/self.toFloat(),self.fractionError())

    def upperBound(self):
        return self.toFloat() + self.toError()

    def lowerBound(self):
        return self.toFloat() - self.toError()

    def __pow__(self, power, modulo=None):
        return errorFloat(pow(self.toFloat(), power), self.fractionError() * power)

    def __truediv__(self, other):
        if type(other) is type(self):
            e=pow(self.fractionError(), 2)
            e+=pow(other.fractionError(), 2)
            return errorFloat(self.__float__() / other.__float__(), np.sqrt(e))
        else:
            e = self.toError()
            e/=other
            return errorFloat(self.__float__() / other.__float__(), e, True)

    def __float__(self):
        return super().__float__()

    def toFloat(self):
        return super().__float__()

    def toError(self):
        return self.error #* np.abs(self.toFloat())

    def fractionError(self):
        if np.abs(self.toFloat())!=0:
            return self.error / np.abs(self.toFloat())
        return 0

    def toErrorFractional(self, error):
        if np.abs(self.toFloat())!=0:
            return error/np.abs(self.toFloat())
        return 0

    def fromErrorFractional(self, error):
            return error*np.abs(self.toFloat())

def toErrorFloat(array, type=None):
    if type is None or type == "normal":
        return errorFloat(np.mean(array), calculateError(array), True)
    elif type == "poisson":
        return errorFloat(np.mean(array), np.sqrt(np.mean(array)), True)

def toErrorFloatSingle(value, error):
    return errorFloat(value, error, True)

def getEngenneringUnit(power):
    switcher = {
        -8: "y",
        -7: "z",
        -6: "a",
        -5: "f",
        -4: "p",
        -3: "n",
        -2: "\u03BC",
        -1: "m",
        0:"",
        1:"k",
        2:"M",
        3:"G",
        4:"T",
        5:"P",
        6:"E",
        7:"Z",
        8:"Y"
    }
    return switcher.get(power, f"({power*3})")

# converter example
# {1: ll.convert}
def loadCSV(path,skip_header=0,converters={}, asStrings=False):
    path = open(path)
    if asStrings:
        return np.genfromtxt(path, delimiter=",", skip_header=skip_header,converters=converters,dtype=None, encoding=None)
    else:
        return np.genfromtxt(path, delimiter=",", skip_header=skip_header,converters=converters,names=True,dtype=None )

def toDataObject(data):
    if not isinstance(data[0], list):
        data = [data]
    if len(data) > 1:
        d = dataObject(title=data[0], data=data[1:])
    else:
        d = dataObject(title=data[0], data=None)
    return d

class dataObject:
    titles = []
    array = []
    def __new__(self, path=None, errorArray=None, **kwargs):
        return super().__new__(self)

    def __add__(self, other):
        if isinstance(other, dataObject):
            a = self.toArray()
            b = other.toArray()
            c = []
            for r in range(len(a)):
                n = []
                for rr in a[r]:
                    n.append(rr)
                for rr in b[r]:
                    n.append(rr)
                c.append(n)
            return toDataObject(c)


    def __init__(self, path=None, errorArray=None, **kwargs):
        if not path is None:
            raw=loadCSV(path, asStrings=True)
            self.titles=list(raw[0])
            self.array=raw[1:]

            temp = []
            for x in range(0, len(self.array)):
                temp.append([])
                for y in range(0, len(self.array[x])):
                    if self.array[x][y] == "":
                        temp[x].append(None)
                    else:
                        try:
                            if errorArray is None:
                                temp[x].append(errorFloat(self.array[x][y], 0))
                            else:
                                temp[x].append(errorFloat(self.array[x][y], errorArray[y], True))
                        except:
                            temp[x].append(self.array[x][y])

            self.array = temp
        else:
            self.titles = kwargs["title"]
            self.array = kwargs["data"]
        return super().__init__()

    def rotate(self):
        ra = rotateArray(self.toArray())
        return dataObject(title=ra[0],data=ra[1:])

    def toArray(self):
        new = self
        if new.array is None:
            return [new.titles]
        t = list(new.asData()).copy()
        t.insert(0, new.titles)
        return t

    def __getitem__(self, index):
        if type(index) is slice:
            if index.stop is None:
                index = slice(index.start, len(self.titles))
            dO = dataObject(title=list(self.titles[index]),data=list(rotateArray(rotateArray(self.asData())[index])))
            return dO
        elif type(index) is tuple:
            new = self[index[0]]
            #r = slice(index[1].start,index[1].stop)
            #print(new.asData()[r][:1])
            t = list(new.asData()).copy()
            t.insert(0, new.titles)

            t = t[index[1]]
            dO = dataObject(title=list(t[0]),data=list(t[1:]))
            return dO
        else:
            if index == 0:
                return self.titles
            return self.asData()[index-1]

    def __delslice__(self, i, j):
        return 0

    def __len__(self):
        return 1+len(self.asData())

    def __str__(self):
        if self.array is None:
            return asTextTable(None, self.titles)
        else:
            return asTextTable(rotateArray(convertArrayToString(self.array)), self.titles)

    def __repr__(self):
        return "dataObject("+")"

    def sumAll(self, type=None):
        p = []
        for d in rotateArray(self.asData()):
            p.append([toErrorFloat(d,type=type)])
        dO = dataObject(title=self.titles,data=rotateArray(p))
        return dO

    def applyErrorToAllData(self, error):
        temp = []
        for x in range(0, len(self.array)):
            temp.append([])
            for y in range(0, len(self.array[x])):
                if self.array[x][y] == "":
                    temp[x].append(None)
                else:
                    try:
                            temp[x].append(errorFloat(self.array[x][y], error, True))
                    except:
                        temp[x].append(self.array[x][y])

        self.array = temp

    def quickPlotLogY(self, show=True, titles=None, formatting='.', **kwargs):
        plt.errorbar(self.titles, rotateArray(self.asData(), True), toErrorArray(rotateArray(self.asData())),
                     toErrorArray(self.titles), formatting, **kwargs)
        plt.yscale('log', nonpositive='clip')
        #plt.semilogy(self.titles, rotateArray(self.asData()), ".", **kwargs)

        if not titles is None:
            plt.xlabel(titles[0])
            plt.ylabel(titles[1])


        if show:
            plt.show()

    def quickPlot(self, show=False, titles=None, formatting='.', **kwargs):
        plt.errorbar(self.titles, rotateArray(self.asData(), True), toErrorArray(rotateArray(self.asData())),
                     toErrorArray(self.titles), formatting, **kwargs)
        #plt.plot(self.titles, rotateArray(self.asData()), ".", **kwargs)

        if not titles is None:
            plt.xlabel(titles[0])
            plt.ylabel(titles[1])

        if show:
            plt.show()

    def asLatexTable(self, caption="temp"):
        return asLatexTable(rotateArray(convertArrayToString(self.array)), self.titles, caption=caption, label=caption.replace(" ", "_"))

    def asData(self):
        return self.array

    def getTitles(self):
        return self.titles

# one two significant figure rule
# If first digit is 1 or 2 : round to 2 s.f.
# Else: round to 1 s.f.
def reducePrecision(num):
    reduced = None
    numS = str(num)
    if numS[0] == "1" or numS[0] == "2":
        reduced = toSignificantFigures(num, 1)
    else:
        reduced = toSignificantFigures(num, 0)
    return reduced

def toSignificantFigures(num, index):
    amount = np.floor(np.log10(np.abs(num.__float__())))
    offset = 10**(amount - index)
    n = num / offset
    r = np.floor(round(n))

    return [r * offset, amount - index]

def toErrorArray(array):
    outarray = []
    for a in array:
        if isinstance(a,list) and len(a) == 1:
            a = a[0]
        if isinstance(a,errorFloat):
            outarray.append(a.toError())
        else:
            outarray.append(0)
    return np.array(outarray)

def loadData(path, errors=None):
    return dataObject(path, errors)

def convertArrayToString(array):
    temp = []
    for x in range(0, len(array)):
        temp.append([])
        for y in range(0, len(array[x])):
            if array[x][y] is None:
                temp[x].append("")
            else:
                temp[x].append(array[x][y].__str__())
    return temp


sym.init_printing(use_unicode=False)

# example:
# expr = ll.expression("x**2")
# expr = expr.evaluate(dict(x=ll.toErrorFloatSingle(3, 5)))
class expression:
    expression = None

    def __new__(self, expressionAsString, evaluate=False):
        return super().__new__(self)

    def __init__(self, expressionAsString, evaluate=False):
        self.expression = sym.parse_expr(expressionAsString, evaluate=evaluate)
        return super().__init__()

    def evaluate(self, subs):
        try:
            hasError = False
            totalError = 0
            for sub in subs:
                if isinstance(subs[sub], errorFloat):
                    hasError = True
                    partial = sym.diff(self.expression, sub).evalf(subs=subs)
                    error = pow(partial*subs[sub].toError(),2).__float__()
                    totalError+=error

            if hasError:
                return toErrorFloatSingle(self.expression.evalf(subs=subs).__float__(), np.sqrt(totalError))
            else:
                return self.expression.evalf(subs=subs).__float__()
        except:
            print(f"\033[91mError: Make sure to sub in values for everything!\nFound variables: {self.expression.free_symbols}\033[0m")
            return

    def __str__(self):
        return self.expression.__str__()

    def pretty(self):
        return sym.pretty(self.expression)

    def __float__(self):
        return expression.__float__()

    def __repr__(self):
        return "dataObject("+")"

def errorPlot(x, y, *argv, **kwargs):
    xErrors=[]
    yErrors=[]

    for tempX in x:
        if isinstance(tempX, errorFloat):
            xErrors.append(tempX.toError())
        else:
            xErrors.append(0)

    for tempY in y:
        if isinstance(tempY, errorFloat):
            yErrors.append(tempY.toError())
        else:
            yErrors.append(0)

    noneAt=[]
    i = 0
    for tempX in x:
        if tempX is None:
            noneAt.append(i)
        i+=1

    i = 0
    for tempY in y:
        if tempY is None:
            noneAt.append(i)
        i+=1

    removed=0
    noneAt = np.sort(np.unique(noneAt))
    for nat in noneAt:
        xErrors.pop(nat-removed)
        yErrors.pop(nat-removed)
        x.pop(nat-removed)
        y.pop(nat-removed)
        removed+=1

    plt.errorbar(x, y, yErrors, xErrors, *argv, **kwargs)
