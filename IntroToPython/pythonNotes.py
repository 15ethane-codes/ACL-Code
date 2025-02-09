print("Hello")

x = 3
y = 4.7
z = "Hi"

print(x, z)

if x < 10:
    x+=1
elif x>10:
    x=-1
else:
    print("x=10")

for i in range(1, 5):
    print(i)

listExample = {"1st value", 6.22, 13, 18, 'yellow'}
print(listExample)
listExample.append('blue')
print(listExample)
print(listExample[0:3])

def add(x, y):
    sum = x+y
    return

print(add(8,5))