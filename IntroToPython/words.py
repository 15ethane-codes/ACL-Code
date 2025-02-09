text = input("Enter some text:\n")
words = text.split()

count = {}

for word in words:
    word = word.lower()
    if word in count:
        count[word] += 1
    else:
        count[word] = 1
print(count)
