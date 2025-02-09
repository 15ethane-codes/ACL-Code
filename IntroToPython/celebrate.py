people = {"Rohit R" : {
            "Birthday" : "9/28/08", "Color" : "Blue", "Dessert" : "Rasmali"},
        "Taha R" : {
            "Birthday" : "6/12/08", "Color" : "Blue", "Dessert" : "Popcorn"},
        "Ethan L" : {
            "Birthday" : "6/20/08", "Color" : "Blue", "Dessert" : "Ice cream"},
        "Aryan P" : {
            "Birthday" : "7/07/08", "Color" : "Black", "Dessert" : "Cake"},
        "Nathan L" : {
            "Birthday" : "7/28/10", "Color" : "Green", "Dessert" : "Pizza"}
          }
print("Let's Celebrate! We have information about:")
for key in people:
    print(key)

name = input("Who would you like to learn about?")

if name in people:
    value = input("What would you like to know? (Birthday, Color, Dessert)")
    info = people[name]
    if value == "Birthday":
        print(name + "'s birthday is " + info[value])
    elif value == "Color":
        print(name +"'s favorite color is " + info[value])
    elif value == "Dessert":
        print(name +"'s favorite dessert is " + info[value])
    else:
        print("Invalid choice. Try again")
else:
    print("Sorry, we don't have that information")