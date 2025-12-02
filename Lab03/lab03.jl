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
function find_omega(fp, gamma)
    omega = 2*π*fp*gamma
    return omega
end

fp = 1.0
gamma = 0.8

omega = find_omega(fp, gamma)
println("Wartośc ω wynosi: ", omega)

# problem 4.3
##
# okresl czy ciagi sa okresowe
using LinearAlgebra

function calculate_period(argument_coefficient)
    period = 1 / argument_coefficient
    if isinteger(period)
        return period
    else
        return "Ciag nie jest okresowy"
    end
end

arg_a = 3π/7
arg_b = 1/8

period_a = calculate_period(arg_a)
period_b = calculate_period(arg_b)

println("okres ciagu x[n] wynosi: ", period_a)
println("okres ciagu g[n] wynosi: ", period_a)

# problem 4.4
##
# jaki warunek dla fg
# Zgodnie z twierdzeniem o próbkowaniu (twierdzenie Nyquista-Shannona)
function fg_find(fg::Real, fs::Real)::Real = fg >= fs ? 1.0 : 0.0

# problem 4.5
##
# interpolate
function interpolate(
    m::AbstractVector,
    s::AbstractVector,
    kernel::Function=sinc
)::Function
    return x -> begin
        sum = 0.0
        Δt = m[2] - m[1]
        for i in eachindex(m)
            sum += s[i] * kernel((x - m[i]) / Δt)
        end
        return sum
    end
end

# problem 4.6
##
# srednia kwadratowa
using CairoMakie

function interpolate(
    m::AbstractVector,
    s::AbstractVector,
    kernel::Function=sinc
)::Function
    return x -> begin
        sum = 0.0
        Δt = m[2] - m[1]
        for i in eachindex(m)
            sum += s[i] * kernel((x - m[i]) / Δt)
        end
        return sum
    end
end

# Funkcja obliczająca błąd średniokwadratowy rekonstrukcji sygnału
function calculate_mse(original::AbstractVector, reconstructed::Function, domain::AbstractVector)
    n = length(domain)
    mse = sum((original[i] - reconstructed(domain[i]))^2 for i in 1:n) / n
    return mse
end

# Funkcja generująca przykładowy sygnał do interpolacji
function generate_signal(frequency::Real, duration::Real, sample_rate::Real)
    t = 0:1/sample_rate:duration
    signal = sin.(2π * frequency * t)
    return t, signal
end

# Parametry eksperymentu
frequency_range = 1:10  # Zakres częstotliwości próbkowania do przetestowania
duration = 1.0          # Czas trwania sygnału
sample_rate = 100.0     # Początkowa częstotliwość próbkowania

# Przeprowadzenie eksperymentu
mse_values = []
for freq in frequency_range
    t, original_signal = generate_signal(5.0, duration, sample_rate)
    m = 0:1/freq:duration
    s = sin.(2π * 5.0 * m)  # Sygnał próbkowany
    reconstructed_signal = interpolate(m, s, sinc)
    mse = calculate_mse(original_signal, reconstructed_signal, t)
    push!(mse_values, mse)
end

# Wykres wyników eksperymentu
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Częstotliwość próbkowania", ylabel="Błąd średniokwadratowy",
    title="Zależność błędu rekonstrukcji od częstotliwości próbkowania")
lines!(ax, frequency_range, mse_values)
fig

# problem 4.7
##
# blad rekonstrukcji
using CairoMakie

# Funkcja obliczająca błąd średniokwadratowy rekonstrukcji sygnału
function calculate_mse_phase(original::AbstractVector, reconstructed::Function, domain::AbstractVector)
    n = length(domain)
    mse = sum((original[i] - reconstructed(domain[i]))^2 for i in 1:n) / n
    return mse
end

# Generowanie sygnału do interpolacji
function generate_signal_phase(frequency::Real, phase::Real, duration::Real, sample_rate::Real)
    t = 0:1/sample_rate:duration
    signal = cos.(π * frequency * t + phase)
    return t, signal
end

# Parametry eksperymentu
phase_range = 0:0.1:2π  # Zakres przesunięcia fazowego
frequency = 5.0          # Częstotliwość sygnału
duration = 1.0           # Czas trwania sygnału
sample_rate = 2 * frequency  # Krytyczna częstotliwość próbkowania

# Przeprowadzenie eksperymentu
mse_values_phase = []
for phase in phase_range
    t, original_signal = generate_signal_phase(frequency, phase, duration, sample_rate)
    m = range(0, stop=duration, length=length(t))
    s = cos.(π * frequency * m + phase)  # Sygnał próbkowany
    reconstructed_signal = interpolate(m, s, sinc)
    mse_phase = calculate_mse_phase(original_signal, reconstructed_signal, t)
    push!(mse_values_phase, mse_phase)
end

# Wykres wyników eksperymentu
fig_phase = Figure()
ax_phase = Axis(fig_phase[1, 1], xlabel="Przesunięcie fazowe", ylabel="Błąd średniokwadratowy",
    title="Zależność błędu rekonstrukcji od przesunięcia fazowego")
lines!(ax_phase, phase_range, mse_values_phase)
fig_phase


# problem 5.1
##
# quantize 
using LinearAlgebra

function quantize(levels::AbstractVector)
    function f(x)
        l_prime = argmin(abs.(x .- levels).^2)
        return levels[l_prime]
    end
    return f
end


# Zdefiniowanie poziomów kwantyzacji
levels = [0.0, 0.5, 1.0]

# Utworzenie funkcji kwantyzującej
f_quantize = quantize(levels)

# Testowanie funkcji kwantyzującej
println(f_quantize(0.2))  # Powinno zwrócić 0.0
println(f_quantize(0.7))  # Powinno zwrócić 0.5
println(f_quantize(0.9))  # Powinno zwrócić 1.0

# problem 5.2
##
# Teoretyczny SQNR
function SQNR(N::Int)
    return 6.02 * N + 1.76
end

N_bits = 12
sqnr = SQNR(N_bits)
println("Teoretyczny stosunek mocy sygnału do mocy szumu kwantyzacji dla $N_bits-bitowego przetwornika wynosi $sqnr dB.")

# problem 5.3
##
# SNR
function SNR(Psignal, Pnoise)
    return 10*log10(Psignal / Pnoise)
end

# problem 5.4
##
# zaleznosc pomiedzy teor. a rzecz. SQNR
using Random

function SQNR(N::Int)
    return 6.02 * N + 1.76
end

function SNR(Psignal, Pnoise)
    return 10*log10(Psignal / Pnoise)
end

function generate_sine_wave(freq::Real, duration::Real, sample_rate::Real)
    t = 0:1 / sample_rate:duration
end

freq = 1000
duration = 1.0
sample_rate = 10000

N_bits = 12

signal = generate_sine_wave(freq, duration, sample_rate)
quantized_signal = round.(Int, signal * (2^(N_bits - 1)))

Psignal = sum(signal.^2) / length(signal)
Pnoise = sum((signal-quantized_signal).^2) / length(signal)

sqnr_theoretical = SQNR(N_bits)
sqnr_actual = SNR(Psignal, Pnoise)

println("Teoretyczny SQNR dla $N_bits-bitowego przetwornika: $sqnr_theoretical dB")
println("Rzeczywisty SQNR dla sygnału sinusoidalnego: $sqnr_actual dB")