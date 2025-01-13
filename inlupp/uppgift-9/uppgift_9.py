# Uppgift 9
# Skapa en funktion is_palindrome(string) som kontrollerar om en given sträng är ett palindrom (dvs. samma framifrån och bakifrån).

def is_palindrome(x: str) -> bool:
   if x== x[::-1]:
       return True
   else:
      return False
print(is_palindrome("radar"))
print(is_palindrome("python"))
print(is_palindrome(""))
