
grades = []

def displayGrades():
    print("The grades are ...")
    print(grades)

def addGrades():
    print("Enter grades")
    while True:
        grade = int(input("Grade: "))
        if grade == -1:
            break
        grades.append(grade)

def getAverageGrade():
    if len(grades) == 0:
        return 0
    return sum(grades) / len(grades)

def roundUpGrades():
    for i in range(len(grades)):
        if grades[i] in (69, 79, 89):
            grades[i] +=1

def removelowGrade():
    if len(grades) > 0:
        grades.remove(min(grades))

#test
addGrades()
roundUpGrades()
removelowGrade()
displayGrades()
print("Average: " + str(getAverageGrade()))

