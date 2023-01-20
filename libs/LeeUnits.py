import LeePhysics as lp

metre = lp.Unit("m")
second = lp.Unit("s")
gram = lp.Unit("g")
ampere = lp.Unit("A")
kelvin = lp.Unit("K")
mole = lp.Unit("mol")
candela = lp.Unit("cd")

kilogram = gram * 1000
newton = lp.CompoundUnit("N", "g*m/(s**2)", dict(g=kilogram, m=metre, s=second))
