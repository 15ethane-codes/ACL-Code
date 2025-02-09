def populate_grades():
    grades = input("Enter the grades separated by commas (e.g., A,B,C,A,B): ")
    grades_list = [grade.strip().upper() for grade in grades.split(',')]
    return grades_list


def count_grades(grades_list):
    count_dict = {}
    for grade in grades_list:
        if grade in count_dict:
            count_dict[grade] += 1
        else:
            count_dict[grade] = 1
    return count_dict


def print_sorted_alphabetically(count_dict):
    sorted_dict = dict(sorted(count_dict.items()))
    print("Sorted alphabetically:")
    '''for grade, count in sorted_dict.items():'''
    print(sorted_dict)
    '''print(f"{grade}: {count}")'''


def print_sorted_reverse_alphabetically(count_dict):
    sorted_dict = dict(sorted(count_dict.items(), reverse=True))
    print("Sorted in reverse alphabetical order:")
    '''for grade, count in sorted_dict.items():'''
    print(sorted_dict)
    '''print(f"{grade}: {count}")'''


def print_sorted_by_value(count_dict):
    sorted_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]))
    print("Sorted by value:")
    '''for grade, count in sorted_dict.items():'''
    print(sorted_dict)
    '''print(f"{grade}: {count}")'''


if __name__ == "__main__":
    grades_list = populate_grades()
    print(grades_list)
    count_dict = count_grades(grades_list)
    print(count_dict)

    print_sorted_alphabetically(count_dict)
    print_sorted_reverse_alphabetically(count_dict)
    print_sorted_by_value(count_dict)

