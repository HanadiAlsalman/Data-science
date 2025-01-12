# Uppgift 2
# Skapa en funktion sum_list(numbers) som returnerar summan av alla siffror i listan.

def sum_list(x: list) -> int:
    sum = 0
    for i in x:
        sum += i
    return sum
print(sum_list([1, 2, 3]))
print(sum_list([]))
print(sum_list([-1, -1, 2]))
print(sum_list([1, 2, 3, 4, 5, 6]))
