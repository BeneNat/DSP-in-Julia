# zadania przewidzane na dzisiaj:
# 7.1.1 - 7.1.3; 7.2.1 - 7.2.7

# problem 7.1.1
##
# problem tablicowy o podanych parametrach
#Δf = 1/(N*Δt)
#Δf to odstęp częstotliwościowy,
#N to liczba punktów w transformacji,
#Δt to odstęp czasu między próbkami.
N = 8192
Δt = 50e-6
Δf = 1/(N*Δt)
println("Odstęp częstotliwościowy między dwoma sąsiadującymi próbkami w wynikowej transformacie: ", Δf, " Hz")

# problem 7.1.2
##
# problem tablicowy o podanych parametrach
f_max = 5000
Δf = 5
N_min = ceil(Int, f_max/Δf)
fs_min = N_min * Δf
println("Minimalna wartość N: ", N_min)
println("Minimalna częstotliwość próbkowania: ", fs_min, " Hz")

# problem 7.1.3
##
# problem tablicowy o podanych parametrach
#f_k = (k*fs)/N
#fs = 1/Δt
#N dlugosc ciagu
arr = [0, 200, 1000, 1600]
Δt = 0.001
N = 2000
#oblcizenie fs
fs = 1/Δt
#@show(arr)
#@show(fs)
ans = []
for i in 1:lastindex(arr)
    push!(ans, (arr[i]*fs)/N)
end
@show(ans)

# problem 7.2.1
##
# funkcja fftfreq
function fftfreq(N::Int, fp::Real)
    return (0:N-1)*fp / N
end

N = 2000
fp = 1000
frequencies = fftfreq(N, fp)
println(frequencies)

# problem 7.2.2
##
# funkcja rfftfreq
function rfftfreq(N::Int, fp::Real)
    return (0:floor(Int, N/2))*fp / N
end

N = 2000
fp = 1000
frequencies = rfftfreq(N, fp)
println(frequencies)

# problem 7.2.3
##
# funkcja amplitude_spectrum
function amplitude_spectrum(x::Vector{Complex{T}}, w::Vector{T}) where T<:Real
    # Sprawdzenie, czy długość sygnału i okna są takie same
    if length(x) != length(w)
        throw(ArgumentError("Długość sygnału i okna musi być taka sama"))
    end
    
    # Obliczenie transformaty Fouriera sygnału z nałożonym oknem
    X = fft(x .* w)
    
    # Obliczenie widma amplitudowego
    A = abs.(X)
    
    return A
end

x = rand(ComplexF64, 100)  # Przykładowy sygnał
w = hamming(100)  # Przykładowe okno (np. okno Hamminga)
A = amplitude_spectrum(x, w)
println(A)

# problem 7.2.4
##
# funkcja power_spectrum
# warto zauwazyc ze to prawie dokladnie amplitude_spectrum
function power_spectrum(x::Vector{Complex{T}}, w::Vector{T}) where T<:Real
    if length(x) != length(w)
        throw(ArgumentError("Długość sygnału i okna musi być taka sama"))
    end
    X = fft(x .* w)
    P = abs2(X)
    return P
end

x = rand(ComplexF64, 100)  # Przykładowy sygnał
w = hamming(100)  # Przykładowe okno (np. okno Hamminga)
P = power_spectrum(x, w)
println(P)

# problem 7.2.5
##
# funkcja psd
function psd(x::Vector{Complex{T}}, w::Vector{T}, fp::Real) where T<:Real
    if length(x) != length(w)
        throw(ArgumentError("Długość sygnału i okna musi być taka sama"))
    end
    P = power_spectrum(x, w)
    N = length(x)
    fs = fp / N
    G = P / fs
    return G
end

x = rand(ComplexF64, 100)  # Przykładowy sygnał
w = hamming(100)  # Przykładowe okno (np. okno Hamminga)
fp = 1000  # Częstotliwość próbkowania w Hz
G = psd(x, w, fp)
println(G)

# problem 7.2.6
##
# funkcja periodogram
function hamming(N::Int)
    w = [0.54 - 0.46 * cos(2π * n / (N - 1)) for n in 0:N-1]
    return w
end
function fft(x)
    N = length(x)
    
    # Jeśli długość sygnału jest równa 1, zwróć go
    if N == 1
        return x
    end
    
    # Jeśli długość sygnału jest parzysta, podziel na dwie części i rekurencyjnie oblicz FFT dla każdej części
    if iseven(N)
        X_even = fft(x[1:2:end])
        X_odd = fft(x[2:2:end])
    else
        # Jeśli długość sygnału jest nieparzysta, dodaj zero do sygnału, aby uczynić go parzystym
        x = [x; 0]
        N = length(x)
        X_even = fft(x[1:2:end])
        X_odd = fft(x[2:2:end])
    end
    
    # Oblicz wartości transformaty Fouriera dla kolejnych części sygnału
    X = similar(x)
    for k = 1:N÷2
        t = X_odd[k] * exp(-2im*π*(k-1)/N)
        X[k] = X_even[k] + t
        X[k + N÷2] = X_even[k] - t
    end
    
    return X
end
function mean(arr::AbstractArray, dims::Int)
    n = size(arr, dims)
    return sum(arr, dims=dims) / n
end

function periodogram(x::Vector{Complex{T}}, K::Int, L::Int, fp::Real) where T<:Real
    N = length(x)
    M = div(N-K, L) + 1
    X_slices = zeros(Complex{T}, K, M)
    w = hamming(K)
    for m in 1:M
        idx_start = 1 + (m - 1) * L
        idx_end = min(N, idx_start + K - 1)
        x_slice = x[idx_start:idx_end]
        x_windowed = x_slice .* w[1:length(x_slice)]
        X_slices[:, m] = fft(x_windowed)
    end
    P_slices = zeros(T, K, M)
    for m in 1:M
        P_slices[:, m] = abs2.(X_slices[:, m])
    end
    G = mean(P_slices, dims=2)[:, 1]
    fs = fp / K
    G = G / fs
    return G
end

x = rand(ComplexF64, 1000)  # Przykładowy sygnał
K = 100  # Długość wycinków
L = 50  # Przesunięcie między wycinkami
fp = 1000  # Częstotliwość próbkowania w Hz
G = periodogram(x, K, L, fp)
println(G)

# problem 7.2.7
##
# sygnal z radia
using WAV
x, fs = wavread("FM.wav")
G = periodogram(x, K, L, fs)