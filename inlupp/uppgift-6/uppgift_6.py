# Uppgift 6
# Skapa en funktion multiplication_table(n, limit) som returnerar multiplikationstabellen för n upp till limit i en lista.


def multiplication_table(n:int, limit:int) -> list:
    multiplication_table = []
    i = 1
    while i <= limit:
        multiplication_table.append(i*n)
        i += 1
    return multiplication_table
print (multiplication_table(2,3))
print (multiplication_table(3,5))