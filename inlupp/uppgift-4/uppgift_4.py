# Uppgift 4
# Skapa en funktion fibonacci(n) som returnerar en lista med de första n Fibonacci-talen.

def f(n: int) -> int:
    if n <= 0:
        return 0 # returnerar 0 för negativa eller noll
    elif n == 1:
        return 1
    else:
        return f(n-1) + f(n-2)

def fibonacci(n: int) -> list:
    fib_list = []
    for i in range(n):
        fib_list.append(f(i))
    return fib_list
print(fibonacci(5))
print(fibonacci(0))
print(fibonacci(1))