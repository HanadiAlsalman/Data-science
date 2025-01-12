# Uppgift 5
# Skapa en funktion filter_odd(numbers) som returnerar en lista med alla jämna tal från den givna listan.

def filter_odd(x: list) -> list:
    jämt_lista = []
    for i in x:
        if i % 2 == 0:
            jämt_lista.append(i)
        else:
            continue
    return jämt_lista
print(filter_odd([1, 2, 3, 4]))
print(filter_odd([1, 3, 5]))
print(filter_odd([]))
