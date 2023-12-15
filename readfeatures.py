def filewrite(name, line):
    f = open(name,"a")
    f.write(line)
    f.close()

for filevalue in range(82):
    path = "feature-targets5/"
    filename = "feature-targets5-"+str(filevalue)+".txt"
    f = open(path + filename, "r")
    for line in f:
        array = line.split(":")
        if(float(array[1])) > 10000:
            filewrite("over10000-targets5/"+filename, line)
    f.close()