import numpy as np

arr = np.array([[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,5]])

for i in range(4):
    arr[i][i] +=3
arr[1:3,:]*=4
print(arr)

student_details = [{'id': 1, 'subject': 'math', 'test1': 70, 'test2': 82},
                   {'id': 2, 'subject': 'math', 'test1': 73, 'test2': 74},
                   {'id': 3, 'subject': 'math', 'test1': 75, 'test2': 86}]

students= []
for student in student_details:
    student['testAvg'] = (student['test1']+student['test2'])/2.0
    del student['test1']
    del student['test2']
    students.append(student)
print(students)

def dotFunction(mA ,mB):
    arr = []
    for i in range(len(mA)):
        row = []
        for j in range (len(mB[0])):
            sum = 0
            for k in range (len(mB)):
                sum+=mA[i][k] * mB[k][j]
            row.append(sum)
        arr.append(row)
    return arr

arrA = np.array([[1,2,3],[2,3,4]])
arrB = np.array([[3,4],[5,6],[6,7]])
print(dotFunction(arrA,arrB))
print(arrA.dot(arrB))
