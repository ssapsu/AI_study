'''
코드 설명: 
1. 13번 돌아가는 for문에서 문자열을 변수 inputStr에 저장
2. 문자열의 앞부분의 정보를 저장할 dataList1과 뒷부분의 정보를 저장할 dataList2를 append함수와 문자열 슬라이싱을 활용하여 리스트에 저장
3. zip함수를 사용하여(list들을 인덱스 순으로 튜플 형식으로 반환해줌; iterable하기에 for문에서 사용 가능) value1과 value2에 누적합을 함
4. 13으로 나누고 출력하기
'''

dataList1 = []; dataList2 = []
value1 = 0; value2 = 0

for i in range(0, 13):
    inputStr = input()
    dataList1.append(inputStr[:2])
    dataList2.append(inputStr[-2:])

for (j, k) in zip(dataList1, dataList2):
    value1 += int(j)
    value2 += int(k)
    
print(str(value1/13), str(value2/13))