def getPage(miniPages):
    a = ""

    for miniPage in miniPages:
        a += miniPage

    return "\\begin{figure}[H]\n" \
           "\\centering\n" \
        + a + \
        "\\end{figure}"

def getMiniPage(graphics):
    a = ""  # "\\includegraphics[width=\\textwidth]{images/force where x and y are near zero/u=0.5.PNG}\n"

    for graphic in graphics:
        a += "\\includegraphics[width=\\textwidth]{" + graphic + "}\n"

    return "\\begin{minipage}{0.333\\textwidth}\n" \
           "\\centering\n" \
        + a + \
        "\\end{minipage}\\hfill\n"


potential = "images/potential/u=zxcdsgb.png"
force = "images/force where x and y are near zero/u=zxcdsgb.png"
changeinforce = "images/change in force/u=zxcdsgb.png"

# graphics between 0 and 1, with 0.1 increments
def getBetween(a, b):
    graphics = [[],[],[]]
    for i in range(a, b):
        graphics[0].append(potential.replace("zxcdsgb", str(i / 10)))
        graphics[1].append(force.replace("zxcdsgb", str(i / 10)))
        graphics[2].append(changeinforce.replace("zxcdsgb", str(i / 10)))


    miniPages = []
    for i in range(0, 3):
        miniPages.append(getMiniPage(graphics[i]))

    page = getPage(miniPages)

    page = page.replace("0.0", "0")
    page = page.replace("1.0", "1")

    return page

page = getBetween(0, 5)
page2 = getBetween(5, 11)
print(page+"\n"+page2)

