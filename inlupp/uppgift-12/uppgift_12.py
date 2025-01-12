# Uppgift 12
# Skapa en funktion create_student_register(students) som tar emot en lista med namn och ålder och returnerar en dictionary där namnet är nyckeln och åldern är värdet.

def create_student_register(x: list) -> dict:
    students = {}
    for namn, ålder in x:
        students[namn] = ålder
    return students 
students = [("Anna", 20), ("Bertil", 25), ("Cecilia", 22)]
print(create_student_register(students))

