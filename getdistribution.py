y_enc_classes = ["Intraoperative rad with other rad before/after surgery",
                "Intraoperative radiation",
                "No radiation and/or cancer-directed surgery",
                "Radiation after surgery",
                "Radiation before and after surgery",
                "Radiation prior to surgery",
                "Sequence unknown, but both were given",
                "Surgery both before and after radiation"
                ]
attributeFieldCount = [0,0,0,0,0,0,0,0]
filename = "2020-extra_removed_preprocessed_target.csv"
file = open(filename, 'r')
count = 0

for line in file:
    if count != 0:
        line = line.strip("\n")
        line = line.split(",")
        print(line[1])
        attributeFieldCount[int(line[1])] += 1
    count += 1
print(count)
print(attributeFieldCount)
sum=0
for i in range(8):
    sum += attributeFieldCount[i]
print(sum)