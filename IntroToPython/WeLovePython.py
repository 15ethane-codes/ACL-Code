# Print
print("Hi")

#Strings
firstN = "Nate"
food = "Burger"

# f is for format
print(f"Hello {firstN}")
print(f"You like {food}")

#Integers
age = 10
print(f"You are {age} years old")

#Float
price = 10.99
print(f"The price is ${10.99}")

#Boolean
is_student = True
if is_student:
    print("You are a student")
else:
    print("not a student")

#Casting
gpa = 4.2
gpa = int(gpa)
print(gpa)

#Input
name = input("Are you cool?: ")
age = int(input("How old are you?: "))
age = age + 1
print(f"{name} is so cool!")
print(f"You are {age} years old")

#Artihmetic/Math
#friends = friends +1
#friends +=1
#friends = friends -2
#friends =-2
#Mod is % and exponent is **

#do some code if some condition is true, othersiw else do something else

#logical operators
temp = 25
is_raining = True

if temp > 35 or temp < 0 or is_raining:
    print("Event cancelled")
else:
    print("Event not cancelled")

