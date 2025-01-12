# Uppgift 3
# Hitta det största talet i en lista

def max_in_list(x: list) -> int:
   max = x[0]
   for i in x:
       if i > max:
           max = i
       else:
         continue
   return max
print(max_in_list([1, 2, 3]))
print(max_in_list([-5, 0, 5]))
print(max_in_list([10]))