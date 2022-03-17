'''
동작 조건: 연산자와 피연산자 사이에 띄어쓰기를 넣어줘야 작동함
코드 설명: 
1. 띄어쓰기를 기반으로 연산자와 피연산자를 구별함
2. 종료 조건 q를 만족시키기 위해 먼저 고려해야 할 대상으로 if elif문의 최상단에 배치
3. 사칙연산의 로직은 띄어쓰기를 기준으로 입력을 split하여 expressionList에 넣어줌
4. expressionList의 1번 인덱스의 값을 기준으로 연산의 종류를 파악함
5. expressionList의 1번 인덱스가 사칙연산 기호 중 한개라면 피연산자가 숫자인지를 판단
6. 위 조건을 만족하지 않을 경우 사용자에게 입력을 다시 할 것을 요청함
7. /의 경우 0으로 나누는 케이스를 예외처리하여 런타임에러를 피함
8. while True로 위 과정을 반복함

'''

def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

while True:
    print('-'*20+"\n\n\n"+"계산식을 입력하시오. (종료: q)")
    
    userInput = input()
    expressionList = []
    
    #! 종료 조건부터
    if userInput == 'q':
        print("종료합니다.\n\n"+'-'*20)
        break
    else:
        expressionList = userInput.split(" ")
        
    
    #@ 입력받았을 떄 연산자를 기준으로 나누기
    
    #* '+'일 때
    if '+' in expressionList[1]:
        if not (isNumber(expressionList[0]) & isNumber(expressionList[2])):
            print("입력을 다시 해주시길 바랍니다.\n\n")
        else:    
            print(expressionList[0] + " " + expressionList[1] + " " + expressionList[2] + " = " + str(float(expressionList[0]) + float(expressionList[2])) + "\n\n")
        
    #* '*'일 때
    elif '*' in expressionList[1]:
        if not (isNumber(expressionList[0]) & isNumber(expressionList[2])):
            print("입력을 다시 해주시길 바랍니다.\n\n")
        else:    
            print(expressionList[0] + " " + expressionList[1] + " " + expressionList[2] + " = " + str(float(expressionList[0]) * float(expressionList[2])) + "\n\n")
        
    #* '/'일 때
    elif '/' in expressionList[1]:
        if not (isNumber(expressionList[0]) & isNumber(expressionList[2])):
            print("입력을 다시 해주시길 바랍니다.\n\n")
            #! 0으로 나눌 때 예외처리를 한다.
        elif int(expressionList[2]) == 0:
            print("0으로 나눌 수 없습니다." + "\n\n")
        else:
            print(expressionList[0] + " " + expressionList[1] + " " + expressionList[2] + " = " + str(float(expressionList[0]) / float(expressionList[2])) + "\n\n")

    #* '-'일 때
    elif '-' in expressionList[1]:
        if not (isNumber(expressionList[0]) & isNumber(expressionList[2])):
            print("입력을 다시 해주시길 바랍니다.\n\n")
        else:    
            print(expressionList[0] + " " + expressionList[1] + " " + expressionList[2] + " = " + str(float(expressionList[0]) - float(expressionList[2])) + "\n\n")
        
    else:
        print("입력을 다시 해주시길 바랍니다.\n\n")