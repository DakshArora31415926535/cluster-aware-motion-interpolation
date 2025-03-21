num1=int(input())
list1=[]
for i in range(0,num1):
    n=int(input())
    list1.append(n)
print(list1)
for i in range(0,len(list1)):
    for j in range(0,len(list1)-i-1):
