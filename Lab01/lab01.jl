##
# skroty klawiszowe:
# run obecna linijka => ctrl + enter
# run aktualna linia i idz do nastepnej => shift + enter
# w VScode podwojny ## oznacza komorke kodu
# run cala komorka alt + enter
# run aktualna komorka i idz do nastepnej komorki => alt + shift + enter
a = 10
b = 20
c = 30

# warto sobie strukturyzowac kod przy uzyciu komorek (##)
##
d = 40
e = 50

#=
Uzywane przez nas paczki:
  [13f3f980] CairoMakie v0.11.9
  [e9467ef8] GLMakie v0.9.9
  [98e50ef6] JuliaFormatter v1.0.54
  [37e2e46d] LinearAlgebra
  [9a3f8284] Random
=#

# w terminalu JUlia "]" przechodzi w tryb menadzera pakietow
# wychodzimy przy uzyciu "backspce" aby wrocic do trybu wykonywania
# przy uzyciu "?" mozemy przejsc w tryb pomocy

# Operacje arytmetyczne
##
a = 1 + 2
b = 4 - 1
c = 2 * 2
d = 8 / 2
e = 7 ÷ 2   # dzielenie bez reszty
f = 2 ^ 8
g = 11 % 2
urojona = im
liczba_pi = pi
liczba_pi_tez = π

# w funkcji moga wystapic argumenty pozycyujne (kolejnosc ma znaczenie) oraz przez slowo kluczowe oddziela sie je ";' (ich kolejnosc nie ma znaczenia)\
# przy wywowlaniu funckji musimy je nazwac (przypisac do wartosci) przyklad:
#=
funkcja:
    function foo(a, b, c=10.0; klucz1, klucz2=1)
        return klucz1 > klucz2 ? a*b*c : a+b+c
    end
    odowlanie:
    foo(1, 2, 3; klucz1=10, klucz2=20)
=#



# problem 1.1 (silnia rekurencyjnie)
##
function silnia_f(n)
    if n<2
        return 1
    else
        return n*silnia_f(n-1)
    end
end
silnia_f(5)

# problem 1.2 (silnia iteracyjnie)
##
silnia = 1
for i in 1:5
    silnia *= i
end
@show silnia    # fajna funkcja do wyswietlania

# problem 1.3 (sprawdzenie czy parzysta)
##
#=
    test = true
    typeof(test)
=#
function is_even(n)
    if (n % 2) == 0
        return true
    else
        return false
    end
end
is_even(15)
is_even(28)

# problem 1.4 (sprawdzenie czy pierwsza)
##
function is_prime(n)
    if n < 2
        return false
    end

    for i in 2:isqrt(n)
        if n % i == 0
            return false
        end
    end
    return true
end
is_prime(66)
is_prime(97)

# problem 1.5 (ciag znakow w odwrotnej kolejnosci)
##
function reverse(s::String)
    l = length(s)
    tab = []
    for i in l:-1:1
        z = s[i]
        push!(tab,z)
    end
    return tab
end

s = "testowy"
@show reverse(s)
# @show reverse(24) # musi byc string
#= d = length(s)
tab = []
for i in d:-1:1
    z = s[i]
    push!(tab, z)
end
@show tab =#


# problem 1.6 (sprawdzenie czy palindrom)
##
function palindrome(a::String)
    l = length(a)
    zgodne = []
    odwrotne = []
    # wynik::Bool
    for i in 1:l
        holder = a[i]
        push!(zgodne, holder)
    end
    for j in l:-1:1
        holder = a[j]
        push!(odwrotne, holder)
    end
    if zgodne == odwrotne
        wynik = true
    else
        wynik = false
    end
    return wynik
end

#= ##
x = [1, 2, "hello"]
y = [1, 2, "Hello"]
if x == y
    wynik = true
else
    wynik = false
end

@show wynik =#
@show palindrome("kajak")
@show palindrome("niekajak")

# problem 1.7 (trojkaty sierpinskiego area)
##
function sierpinski_area(N)
    # N to rzad trojkata sierpinskiego
    if N == 0
        return 1.0 # pole trojkata sierpinskiego rzedu 0 wynosi 1
    else
        sub_triangle_area = sierpinski_area(N-1) / 4
        return 3 * sqrt(3) / 2 * (sub_triangle_area * 3)
    end
end

@show sierpinski_area(4)

# problem 1.8 (pierwiastek kwadratowy metoda Newtona)
##
function sqrt_newton(a::Real; x0=1.0, tol=1e-6, max_iter=1000)
    f(X) = x^2 - a
    df(x) = 2x
    x = x0
    for i in 1:max_iter
        x_next = x - f(x) / df(x)
        if abs(x_next - x) < tol
            return x_next
        end
        x = x_next
    end
    return x
end

a_values = [0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 2137.0]
for a in a_values
    println("Pierwiastek kwadratowy z $a: ", sqrt_newton(a))
end

# problem 1.9 (zbieznosc)
##
# wersja 2 (nowsza)
using Plots  # Importuj bibliotekę do tworzenia wykresów

function iterate_sequence(p, max_iter)
    z = 0.0
    for n in 1:max_iter
        z = z^2 + p
        if abs(z) >= 2
            return n
        end
    end
    return max_iter
end

function calculate_max_convergence(p_real_range, p_imag_range, max_iter)
    convergence_matrix = zeros(length(p_imag_range), length(p_real_range))
    for (j, imag) in enumerate(p_imag_range)
        for (i, real) in enumerate(p_real_range)
            p = real + imag*im
            convergence_matrix[j, i] = iterate_sequence(p, max_iter)
        end
    end
    return convergence_matrix
end

# Definiuj zakresy dla części rzeczywistej i urojonej p
p_real_range = range(-1, 2, length=500)
p_imag_range = range(-1, 1, length=500)

# Oblicz maksymalną zbieżność dla każdego punktu w zakresie
max_iter = 1000
convergence_matrix = calculate_max_convergence(p_real_range, p_imag_range, max_iter)

# Wykreśl macierz zbieżności
heatmap(p_real_range, p_imag_range, log10.(convergence_matrix),  # Using log10 for logarithmic scale
    xlabel="Real(p)", ylabel="Imag(p)", color=:viridis,
    title="Max Convergence (K) for Sequence z_n+1 = z_n^2 + p",
    clim=(0, log10(max_iter)), colorbar_title="Max Iterations (K)")


# wersja 1
#= using LinearAlgebra
using BenchmarkTools
using Plots

function k_convergence(p, max_iter)
    z = 0
    for n in max_iter
        z = z^2 + p
        if abs(z) > 2
            return n - 1 # Zwróć ilość iteracji do momentu, gdy |z| >= 2
        end
    end
    return max_iter # Zwróć maksymalną ilość iteracji, jeśli ciąg nie osiągnął |z| >= 2
end

# zakres punktow p do badania
real_range = -1.0:0.01:1.99
imag_range = -1:0.01:0.99

# Pusta macierz przechowująca wyniki K-zbieżności dla każdego punktu p
max_K = zeros(length(real_range), length(imag_range))

for (i, real_part) in enumerate(real_range)
    for (j, imag_part) in enumerate(imag_range)
        p = real_part + imag_part*im
        max_K[i, j] = k_convergence(p, 1000)
    end
end

# wykres
heatmap(real_range, imag_range, max_K', color=:viridis, xlabel="Real part", ylabel="Imaginary part", c=:blues)
 =#

# problem 1.10 (cykliczne liczby pierwsze ponizej miliona)
##
function cyclic_permutations(n)
    n_str = string(n)
    len = length(n_str)
    permutations = [parse(Int, n_str[i:end] * n_str[1:i-1]) for i in 1:len]
    return permutations
end

function is_prime(n)
    if n < 2
        return false
    end

    for i in 2:isqrt(n)
        if n % i == 0
            return false
        end
    end
    return true
end

function is_circular_prime(n)
    for perm in cyclic_permutations(n)
        if !is_prime(perm)
            return false
        end
    end
    return true
end

function count_circular_primes_below_limit(limit)
    count = 0
    for p in 2:limit
        if is_prime(p) && is_circular_prime(p)
            count += 1
        end
    end
    return count
end

limit = 1_000_000
count = count_circular_primes_below_limit(limit)
println("Ilość cyklicznych liczb pierwszych poniżej $limit: ", count)