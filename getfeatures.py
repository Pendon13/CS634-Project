arrayFeature = []
for i in range(256):
    arrayFeature.append(int(0))

def getFeature(number):
    filevalue = number
    path = "over10000-targets5/"
    filename = "feature-targets5-"+str(filevalue)+".txt"

    f = open(path+filename, 'r')
    for line in f:
        split = line.split(" ")
        value = int(split[1].strip(":"))+5
        arrayFeature[value] += 1

for i in range(82):
    getFeature(i)
print(arrayFeature)