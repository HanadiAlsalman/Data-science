# Uppgift 6
# Skapa en funktion multiplication_table(n, limit) som returnerar multiplikationstabellen för n upp till limit i en lista.

def multiplication_tabellen(n:int, limit:int) -> list:
    multiplication_tabellen = []
    for i in range (1, limit+1):
        multiplication_tabellen.append(i*n)
    return multiplication_tabellen
print (multiplication_tabellen(2,3))
print (multiplication_tabellen(3,5))


def multiplication_tabellen(n:int, limit:int) -> list:
    multiplication_tabellen = []
    i = 1
    while i <= limit:
        multiplication_tabellen.append(i*n)
        i += 1
    return multiplication_tabellen
print (multiplication_tabellen(2,3))
print (multiplication_tabellen(3,5))