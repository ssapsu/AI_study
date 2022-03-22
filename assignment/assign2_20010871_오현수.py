class MyStack():
    stackList = []
    
    def push(self, int):
            self.stackList.append(int)
        
    def pop(self):
        if len(self.stackList)>0:
            print(str(self.stackList.pop()))
        else:
            print(str(-1))
            
    def size(self):
        print(str(len(self.stackList)))
        
    def empty(self):
        if len(self.stackList) == 0:
            print(str(1))
        else:
            print(str(0))
            
    def top(self):
        if len(self.stackList) > 0:
            print(self.stackList[len(self.stackList)-1])
        else:
            print(str(-1))
    
    def clear(self):
        del self.stackList[:]

stack1 = MyStack()
for i in range(int(input())):
    userInput = input().split(' ')
    if userInput[0] == 'push':
        stack1.push(userInput[1])
    elif userInput[0] == 'pop':
        stack1.pop()
    elif userInput[0] =='size':
        stack1.size()
    elif userInput[0] =='empty':
        stack1.empty()
    elif userInput[0] =='top':
        stack1.top()
    elif userInput[0] =='clear':
        stack1.clear()
