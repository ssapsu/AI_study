dataList1 = []; dataList2 = []
value1 = 0
value2 = 0

for i in range(0, 13):
    inputStr = input()
    dataList1.append(inputStr[:2])
    dataList2.append(inputStr[-2:])

for i, (j, k) in enumerate(zip(dataList1, dataList2), start = 1):
    value1 += int(j)
    value2 += int(k)
    
print(str(value1/13), str(value2/13))