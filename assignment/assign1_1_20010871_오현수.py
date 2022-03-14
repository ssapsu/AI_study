from curses.ascii import isdigit

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
        if not (expressionList[0].isdigit() & expressionList[2].isdigit()):
            print("입력을 다시 해주시길 바랍니다.\n\n")
        else:    
            print(expressionList[0] + expressionList[1] + expressionList[2] + " = " + str(int(expressionList[0]) + int(expressionList[2])) + "\n\n")
        
    #* '*'일 때
    elif '*' in expressionList[1]:
        if not (expressionList[0].isdigit() & expressionList[2].isdigit()):
            print("입력을 다시 해주시길 바랍니다.\n\n")
        else:    
            print(expressionList[0] + expressionList[1] + expressionList[2] + " = " + str(int(expressionList[0]) * int(expressionList[2])) + "\n\n")
        
    #* '/'일 때
    elif '/' in expressionList[1]:
        if not (expressionList[0].isdigit() & expressionList[2].isdigit()):
            print("입력을 다시 해주시길 바랍니다.\n\n")
            #! 0으로 나눌 때 예외처리를 한다.
        elif int(expressionList[2]) == 0:
            print("0으로 나눌 수 없습니다." + "\n\n")
        else:
            print(expressionList[0] + expressionList[1] + expressionList[2] + " = " + str(int(expressionList[0]) / int(expressionList[2])) + "\n\n")

    #* '-'일 때
    elif '-' in expressionList[1]:
        if not (expressionList[0].isdigit() & expressionList[2].isdigit()):
            print("입력을 다시 해주시길 바랍니다.\n\n")
        else:    
            print(expressionList[0] + expressionList[1] + expressionList[2] + " = " + str(int(expressionList[0]) - int(expressionList[2])) + "\n\n")
        
    else:
        print("입력을 다시 해주시길 바랍니다.\n\n")