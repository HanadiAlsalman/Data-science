# Uppgift 8
# Skapa en funktion count_letters(string) som returnerar en dictionary med varje bokstav som nyckel och antalet förekomster som värde.

def count_letters(x: str) -> dict:
    count_letters = {}
    x= x.lower() # gör alla bokstäver till små
    for i in x:
        if i == " ": # om det finns mellanslag hoppa över
            continue
        if i in count_letters:
            count_letters[i] += 1
        else:
            count_letters[i] = 1
    return count_letters
print(count_letters("hello"))
print(count_letters(""))
print(count_letters("aaa"))