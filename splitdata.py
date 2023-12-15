def filewrite(name, line):
    f = open(name,"a")
    f.write(line)
    f.close()

filesplit = 82
file = open("targets5.csv","r")
print("starting")
for i, line in enumerate(file):
    if i == 0:
        for j in range(filesplit):
            filewrite("./splitdata-targets5/targets5-"+str(j)+".txt", line)
        continue
    for j in range(filesplit):
        if i % filesplit == j:
            print(i)
            filewrite("./splitdata-targets5/targets5-"+str(j)+".txt", line)
    

print("finished")
file.close()
