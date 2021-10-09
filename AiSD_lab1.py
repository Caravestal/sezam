print("zad.1")
i = "M"
n = "Horabik"
def imie(x,y):
    return i + "." + n
print(imie(i,n))

print("zad.2")
i = "marta"
n = "horabik"
def im(x,y):
    return i.capitalize() + "." + n.capitalize()
print(im(i,n))

print("zad.3")
def uro(a,b,c):
    temp = str(a) + str(b)
    rok = int(temp)
    wyn = rok - c
    return wyn
print(uro(20,21,21))

print("zad.4")
def im(x,y):
    return x.capitalize() + "." + y.capitalize()
i = "marta"
n = "horabik"
def z(a,b,fun):
    return fun(a,b)
print(z(i,n,im))

print("zad.5")
def w(x,y):
    wyn = x/y
    print(wyn)
    if x>0 and y>0 and y!=0:
        return "Obie liczby sa dodatnie i ruga liczba nie rowna sie 0"
print(w(6,3))

print("zad.6")
wyn = 0
while wyn <= 100:
    a = int(input())
    wyn += a
    print(wyn)

print("zad.7")
def fun(list):
    return tuple(list)
print(fun([1,2,3]))

print("zad.8")
lista = []
while len(lista) < 6:
    lista.append(input())
tuple(lista)
print(lista)

print("zad.9")
def tydz(a):
    t = {1: "Poniedzialek", 2: "Wtorek", 3: "Sroda", 4: "Czwartek", 5: "Piatek", 6: "Sobota", 7: "Niedziela"}
    return t[a]
print(tydz(4))

print("zad.10")
def palindrom(a):
    b = a[::-1]
    if a == b:
        return "jest to palindrom"
    else:
        return "to nie jest palindrom"
print(palindrom("alala"))
