# read file
def readText(filePath):
    try:
        f = open(filePath, "r")
        str = f.read()
        return str
    finally:
        if f:
            f.close()


str = readText("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.pts")

def textToPoint(text):
    lines = text.split("\n")
    print(len(lines))
    a = []
    for i, val in enumerate(lines):
        print("序号：%s   值：%s" % (i + 1, val))
        if(i<3):
            continue
        if(i>70):
            continue
        print("append：%s   值：%s" % (i + 1, val))
        vals = val.split(" ")
        a.append((float(vals[0]),float(vals[1])))

    return a

a = textToPoint(str)
for val in a:
    print(val[0]+val[1])