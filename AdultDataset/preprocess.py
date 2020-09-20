import csv

filePath = "AdultDataset/adult.data"

rList = []
with open("AdultDataset/adult-strip.data", 'w', newline="") as r:
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            # rList.append([x.strip() for x in row])
            writer = csv.writer(r)
            writer.writerow([x.strip() for x in row])

