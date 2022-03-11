while True:
    print('-'*20+"\n\n\n"+"계산식을 입력하시오. (종료: q)")
    
    userInput = input()
    operandList = []
    
    #! 종료 조건부터
    if userInput == 'q':
        print("종료합니다.\n\n"+'-'*20)
        break
    
    #@ 입력받았을 떄 연산자를 기준으로 나누기
    
    #* '+'일 때
    if '+' in userInput:
        operandList = userInput.split('+')
        print(operandList[0] + "+" + operandList[1] + " = " + str(int(operandList[0]) + int(operandList[1])) + "\n\n")
        
    #* '*'일 때
    elif '*' in userInput:
        operandList = userInput.split('*')
        print(operandList[0] + "*" + operandList[1] + " = " + str(int(operandList[0]) * int(operandList[1])) + "\n\n")
        
    #* '/'일 때
    elif '/' in userInput:
        operandList = userInput.split('/')
        #! 0으로 나눌 때 예외처리를 한다.
        if int(operandList[0]) == 0:
            print("0으로 나눌 수 없습니다." + "\n\n")
        else:
            print(operandList[0] + "/" + operandList[1] + " = " + str(int(operandList[0]) / int(operandList[1])) + "\n\n")

    #* '-'일 때
    elif '-' in userInput:
        operandList = userInput.split('-')
        print(operandList[0] + "-" + operandList[1] + " = " + str(int(operandList[0]) - int(operandList[1])) + "\n\n")
        
    