# Uppgift 10
# Skapa en funktion celsius_to_fahrenheit(celsius) som konverterar en temperatur från Celsius till Fahrenheit.

def celsicus_to_fahrenheit(c: float) -> float:
    f= (c) * 1.8 + 32
    return f
print(celsicus_to_fahrenheit(0))
print(celsicus_to_fahrenheit(100))
print(celsicus_to_fahrenheit(-40))
