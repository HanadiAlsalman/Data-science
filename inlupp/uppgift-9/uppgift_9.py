# Uppgift 9
# Skapa en funktion is_palindrome(string) som kontrollerar om en given sträng är ett palindrom (dvs. samma framifrån och bakifrån).

def is_palimdrome(x: str) -> bool:
   if x== x[::-1]:
       return True
   else:
      return False
print(is_palimdrome("radar"))
print(is_palimdrome("python"))
print(is_palimdrome(""))
