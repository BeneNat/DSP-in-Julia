# zadania na dzisiaj:
# 4.1 - 4.4, 4.5 - 4.7, 5.1 - 5.4

# problem 4.1
##
# wynikowy ciag podanego sygnalu
using CairoMakie
f = 250
dt = 1/f
t_start = 0
t_stop = 1
s = 256
t = range(;start=t_start, step=dt, length = s)
x = sin.(200*π.*t)
lines(t, x)
y = x
@show y

# problem 4.2
##
# znajdz omega

# problem 4.3
##
# okresl czy ciagi sa okresowe

# problem 4.4
##
# jaki warunek dla fg

# problem 4.5
##
# interpolate

# problem 4.6
##
# srednia kwadratowa

# problem 4.7
##
# blad rekonstrukcji

# problem 5.1
##
# quantize 

# problem 5.2
##
# Teoretyczny SQNR

# problem 5.3
##
# SNR

# problem 5.4
##
# zaleznosc pomiedzy teor. a rzecz. SQNR
