import cPickle

f = open('t')
whs = cPickle.load(f)

num = len(whs) - 10 + 1
min_distance = 10000
distance_list = []
dt = {}
for i in range(num):
    distance = 0
    for j in range(i, i + 9):
        distance += whs[j + 1][0] - whs[j][0]
    dt[distance] = whs[i:i + 10]
    distance_list.append(distance)
    min_distance = min(distance, min_distance)

distance_list.sort()
print distance_list
print "----------"
print min_distance