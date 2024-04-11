# zeby uzywac naszego modulu
# include("CPS.jl)
# zeby uzywac funkcji z modulu
# CPS.dft()

# zadania na dzisiaj:
# 2.1, 2.2, 3.1-3.5, 2.7, 2.10, 2.11, 2.14, 2.15, 2.22, 2.23, 2.26



# problem 2.1
##
# rzeczy. syg. harmoniczny
using Plots

# Określenie parametrów sygnału
amplituda = 2
czestotliwosc = 25  # w Hz
przesuniecie_fazowe = π/4  # w radianach
czas_poczatkowy = 0.25  # w sekundach
czestotliwosc_probkowania = 1000  # w próbkach na sekundę

# Obliczenie czasu trwania sygnału
czas_trwania = 1  # w sekundach

# Obliczenie liczby próbek
liczba_probek = round(Int, czas_trwania * czestotliwosc_probkowania)

# Generowanie wektora czasu
wektor_czasu = collect(range(0, czas_trwania, length=liczba_probek))

# Obliczenie wartości sygnału harmonicznego
sygnal_harmoniczny = amplituda * sin.(2π * czestotliwosc * wektor_czasu .+ przesuniecie_fazowe)

# Wyświetlenie wykresu
plot(wektor_czasu, sygnal_harmoniczny, xlabel="Czas [s]", ylabel="Amplituda", label="Sygnał harmonicznego", legend=:bottomright)

# problem 2.2
##
# zesp. syg. harmoniczny
amplituda = 0.25
czestotliwosc = π/2
przesuniecie_fazowe = π
czas_poczatkowy = 5
czas_koncowy = 10
czestotliwosc_probkowania = 2048

liczba_probek = round(Int, (czas_koncowy - czas_poczatkowy) * czestotliwosc_probkowania)

wektor_czasu = collect(range(czas_poczatkowy, czas_koncowy, length=liczba_probek))
sygnal_harmoniczny = amplituda * exp.(2π * czestotliwosc * im * wektor_czasu .+ przesuniecie_fazowe)

plot(wektor_czasu, real.(sygnal_harmoniczny), xlabel="Czas [s]", ylabel="Amplituda", label="Re(x)", legend=:bottomright)
plot!(wektor_czasu, imag.(sygnal_harmoniczny), xlabel="Czas [s]", ylabel="Amplituda", label="Im(x)")

# problem 3.1
##
# wartosc srednia sygnalu dyskretnego
function mean(x)
    srednia = sum(x)/length(x)
    return srednia
end

# problem 3.2
##
# wart. miedzyszczytowa syg. dysk.
function peak2peak(x)
    # normalne
    miedzyszczytowa = max(x) - min(x)
    return miedzyszczytowa
    # co byloby gdyby obydwie wart byly ujemne
    # to samo bo max = -1 min = -4 to bedzie -1 - (-4) = 3
end

#problem 3.3
##
# energia sygn. dysk.
function energy()
    energia = sum(x.*x)
    return energia
end

# problem 3.4
##
# moc syg. dysk.
function power(x)
    moc = sum(x.*x)/length(x)
    return moc
end

# problem 3.5
##
# wart skuteczna syg. dysk.
function rms(x)
    # wart. skuteczna to pierwiastek mocy
    # moznaby uzyc wczesniejszej funkcji ale teraz bedzie niezalezna
    moc = sum(x.*x)/length(x)
    skuteczna = sqrt(moc)
    return skuteczna
end

# problem 2.7
##
# impuls przypominajacy literke "M" o szerokosci T w chwili t
using Plots

function cw_literka_M(t, T)

end

# problem 2.10
##
# wart. okresowego syg. fali piloksztaltenej z opad. zboczem w chwili t
using Plots

function sawtooth_wave(t)
    # Obliczamy wartość fali piłokształtnej w chwili t
    t_mod = t % 1  # Modulo 1, aby zachować okresowość sygnału
    return 2 * (0.5 - t_mod)
end

# Przygotowanie danych do wykresu
czas_trwania = 2.0  # Trwający czas
czestotliwosc_probkowania = 1000  # Próbkowanie co 1 ms
liczba_probek = round(Int, czas_trwania * czestotliwosc_probkowania)
wektor_czasu = collect(range(0, czas_trwania, length=liczba_probek))
sygnal_piłokształtny = [sawtooth_wave(t) for t in wektor_czasu]

# Wykres
plot(wektor_czasu, sygnal_piłokształtny, xlabel="Czas", ylabel="Amplituda", label="Fala piłokształtna", legend=:bottomright)


# problem 2.11
##
# okr. syg. fali. trojkatnej w chwili t
using Plots

function triangular_wave(t)
    # Obliczamy wartość fali trójkątnej w chwili t
    t_mod = t % 1  # Modulo 1, aby zachować okresowość sygnału
    if t_mod < 0.5
        return 4 * t_mod - 1
    else
        return -4 * t_mod + 3
    end
end

# Przygotowanie danych do wykresu
czas_trwania = 2.0  # Trwający czas
czestotliwosc_probkowania = 1000  # Próbkowanie co 1 ms
liczba_probek = round(Int, czas_trwania * czestotliwosc_probkowania)
wektor_czasu = collect(range(0, czas_trwania, length=liczba_probek))
sygnal_trójkątny = [triangular_wave(t) for t in wektor_czasu]

# Wykres
plot(wektor_czasu, sygnal_trójkątny, xlabel="Czas", ylabel="Amplituda", label="Fala trójkątna", legend=:bottomright)


# problem 2.14
##
# impulse reapeter
function impulse_reapeter()
    
end

# problem 2.15
##
# wart. okresowego syg. fali piloksztaltenej z narast. zboczem w chwili t
function ramp_wave()
    
end

# problem 2.22
##
# wart. dysk. impulsu jednostkowego (Kronecker)
function kronecker(n)
    if n == 0
        return 1
    else
        return 0
    end
end

## problem 2.23
##
# wart. dysk. impulsu jednostkowego (funkcja skokowa Heaviside’a)
function heaviside(n)
    if n < 0
        return 0
    else
        return 1
    end
end

# problem 2.26
##
# wektor z probkami dysk. okna Hanninga
function hanning(N)
    
end

##
using Plots
