# equations is stored locally in the file equations.txt
path = "equations.txt"
equations = open(path, "r").read()

# file is in windows-1252
equations = equations.encode("utf-8").decode("windows-1252")

output = ""

# for each line in the file equations.txt
for line in equations.splitlines():
    #if the line is empty, skip it
    if line == "":
        continue

    # if the line doesn't contain an equals, do not add math mode
    if not "=" in line:
        output += line
        output += "\n"
        continue

    # add latex equations begin and end to the line
    output += "$$" + line + "$$"

    # add a new line
    output += "\n"

# write the output to the file equations new.txt
open("equations new.txt", "w").write(output)

