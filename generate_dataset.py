import random
l=[]
with open("D:/image captions/captions.txt","r") as f:
	doc=f.read()

for line in doc.split('\n'):
	if len(line) < 1:
			continue
	identifier = line.split('.')[0]
	l.append(identifier)
l=set(l)
l=list(l)
random.shuffle(l)

with open("train.txt","a") as f:
	for i in range(6000):
		f.write(l[i]+"\n")

with open("validate.txt","a") as f:
	for i in range(6000,7000):
		f.write(l[i]+"\n")

with open("test.txt","a") as f:
	for i in range(7000,8091):
		f.write(l[i]+"\n")
