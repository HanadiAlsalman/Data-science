# Uppgift 7
# Skapa en funktion validate_password(password) som kontrollerar att lösenordet är minst 8 tecken långt och innehåller minst en siffra.

def validate_password(x: str) -> bool:
    if len(x) < 8:
        return False
    for i in x:
        if i in "0123456789":
            return True
    return False
print(validate_password("abc12345"))
print(validate_password("short"))
print(validate_password("password1"))