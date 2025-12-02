# zadania na dzisiaj:
# 8.1 - 8.8
# 8.1.1 - 8.1.8
# systemy liniowe

# problem 8.1
##
# czy system jest liniowy i niezmienny
# Funkcja sprawdzająca liniowość systemu
x₁ = [1, 2, 3, 4, 5]
x₂ = [2, 3, 4, 5, 6]
a = 2
b = 3
shifted = vcat(0, x₁[1:end-1])

system_a(x) = 2 .* x .+ 3
system_b(x) = x .* sin(2π/7 + π/6)
system_c(x) = x .^ 2
system_d(x) = cumsum(x)

systems = [system_a, system_b, system_c, system_d]

for (i, system) in enumerate(systems)
    println("System $i:")
    # spr liniowosci
    y₁ = system(x₁)
    y₂ = system(x₂)
    y_combined = (a .* x₁ .+ b .* x₂)
    y_expected = a .* y₁ .+ b .* x₂
    linear = y_combined == y_expected
    println("  Liniowy: ", linear)
    #spr niezm w czasie
    y = system(x₁)
    y_shifted = system(shifted)
    y_expected_shifted = vcat(0, y[1:end-1])
    time_invariant = y_shifted == y_expected_shifted
    println("  Niezmienny w czasie: ", time_invariant)
end

# problem 8.2
##
# wyznacz ciag wyj

δ(n) = n == 0 ? 1 : 0
u(n) = n >= 0 ? 1 : 0

x_a(n) = δ(n) + 2 * δ(n - 1) + δ(n - 2)

h_a(n) = u(n)

x_b(n) = δ(n + 2) + 2 * δ(n + 1) + δ(n) + δ(n - 1) + δ(n - 2)

h_b(n) = δ(n + 2)

x_c(n, α) = α^n * u(n)

h_c(n, β) = β^n * u(n)

x_d(n) = u(n)

h_d(n) = δ(n - 2) - δ(n - 3)

function convolution(x, h, range)
    y = zeros(length(range))
    for i in eachindex(range)
        n = range[i]
        for k in range
            y[i] += x(k) * h(n-k)
        end
    end
    return y
end

range_a = 0:10
range_b = -5:5
range_c = 0:10
range_d = 0:10

println("Przypadek a:")
y_a = convolution(x_a, h_a, range_a)
for i in eachindex(range_a)
    println("y[", range_a[i], "] = ", y_a[i])
end

println("\nPrzypadek b:")
y_b = convolution(x_b, h_b, range_b)
for i in eachindex(range_b)
    println("y[", range_b[i], "] = ", y_b[i])
end

println("\nPrzypadek c:")
α = 0.5
β = 0.3
y_c = convolution(n -> x_c(n, α), n -> h_c(n, β), range_c)
for i in eachindex(range_c)
    println("y[", range_c[i], "] = ", y_c[i])
end

println("\nPrzypadek d:")
y_d = convolution(x_d, h_d, range_d)
for i in eachindex(range_d)
    println("y[", range_d[i], "] = ", y_d[i])
end

# problem 8.3
##
# czy system jest stabilny
h_a(n) = n == -2 ? 1 : 0
h_b(n) = n >= 0 ? (1/2)^n : 0
h_c(n) = n <= 0 ? 2^n : 0

function check_stability(h)
    sum = 0
    n = 0
    while true
        val = abs(h(n))
        if val ==0
            break
        end
        sum += val
        n += 1
    end
    return sum < Inf
end

println("system a jest stabilny: ", check_stability(h_a))
println("system b jest stabilny: ", check_stability(h_b))
println("system c jest stabilny: ", check_stability(h_c))

# problem 8.4
##
# odwrotna z transformacja
# Funkcja obliczająca odwrotną Z-transformację


# problem 8.5
##
# Fibonacci


# problem 8.6
##
# wyznacz h2[n]


# problem 8.7
##
# funkcja wlasna

# problem 8.8
##
# rozwaz system

# Problemy implementacyjne

# problem 8.1.1
##
# funckja conv
function conv(f::Vector, g::Vector)::Vector
    N = length(f)
    M = length(g)
    result = zeros(Float64, N+M-1)
    for n in 1:(N+M-1)
        for k in 1:M
            if 1 <= n - k + 1 <= N
                result[n] += f[n-k+1] * g[k]
            end            
        end
    end
    return result
end

# Przykładowe użycie:
f = [1.0, 2.0, 3.0]
g = [0.0, 1.0, 0.5]
conv_result = conv(f, g)
println(conv_result)

# problem 8.1.2
##
# funkcja fast_conv
# Implementacja FFT
# Implementacja FFT
function fft(x::Vector{Complex{Float64}})::Vector{Complex{Float64}}
    N = length(x)
    if N <= 1
        return x
    end
    even = fft(x[1:2:end])
    odd = fft(x[2:2:end])
    T = [exp(-2im * π * k / N) * odd[k] for k in 1:(div(N, 2))]
    return vcat([even[k] + T[k] for k in 1:(div(N, 2))], [even[k] - T[k] for k in 1:(div(N, 2))])
end

# Implementacja IFFT
function ifft(x::Vector{Complex{Float64}})::Vector{Complex{Float64}}
    N = length(x)
    x_conj = conj.(x)
    x_ifft = fft(x_conj)
    return conj.(x_ifft) / N
end

# Funkcja fast_conv
function fast_conv(f::Vector{Float64}, g::Vector{Float64})::Vector{Float64}
    N = length(f)
    M = length(g)
    size = N + M - 1
    f_padded = vcat(f, zeros(size - N))
    g_padded = vcat(g, zeros(size - M))
    F_f = fft(complex(f_padded))
    F_g = fft(complex(g_padded))
    F_fg = F_f .* F_g
    result = ifft(F_fg)
    return real(result)
end

# Przykładowe użycie
f = [1.0, 2.0, 3.0]
g = [0.0, 1.0, 0.5]
fast_conv_result = fast_conv(f, g)
println(fast_conv_result)

# problem 8.1.3
##
# funkcja overlap_add
# Implementacja FFT
function fft(x::Vector{Complex{Float64}})::Vector{Complex{Float64}}
    N = length(x)
    if N <= 1
        return x
    end
    even = fft(x[1:2:end])
    odd = fft(x[2:2:end])
    T = [exp(-2im * π * k / N) * odd[k] for k in 1:(div(N, 2))]
    return vcat([even[k] + T[k] for k in 1:(div(N, 2))], [even[k] - T[k] for k in 1:(div(N, 2))])
end

# Implementacja IFFT
function ifft(x::Vector{Complex{Float64}})::Vector{Complex{Float64}}
    N = length(x)
    x_conj = conj.(x)
    x_ifft = fft(x_conj)
    return conj.(x_ifft) / N
end

# Funkcja overlap_add
function overlap_add(f::Vector{Float64}, g::Vector{Float64}, L::Int)::Vector{Float64}
    N = length(f)
    M = length(g)
    P = N + M - 1
    result = zeros(Float64, P)
    g_padded = vcat(g, zeros(L - M))
    G = fft(complex(g_padded))
    
    for k in 1:L:N
        f_segment = f[k:min(k+L-1, N)]
        f_segment_padded = vcat(f_segment, zeros(L - length(f_segment)))
        F_segment = fft(complex(f_segment_padded))
        conv_segment = ifft(F_segment .* G)
        segment_length = min(L + M - 1, P - k + 1)
        result[k:k + segment_length - 1] += real(conv_segment[1:segment_length])
    end
    
    return result
end

# Przykładowe użycie
f = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
g = [1.0, -1.0, 1.0]
L = 4
overlap_add_result = overlap_add(f, g, L)
println(overlap_add_result)



# problem 8.1.4
##
# funckja overlap_save
function overlap_save(f::Vector{Float64}, g::Vector{Float64}, L::Int)::Vector{Float64}
    N = length(f)
    M = length(g)
    P = N + M - 1
    result = zeros(Float64, P)
    G = fft(complex(vcat(g, zeros(L - M))))
    
    segment_start = 1
    while segment_start <= N
        segment_end = min(segment_start + L - 1, N)
        segment = f[max(1, segment_start - M + 1):segment_end]
        segment_padded = vcat(zeros(L - length(segment)), segment)
        F_segment = fft(complex(segment_padded))
        conv_segment = ifft(F_segment .* G)
        segment_length = min(L, P - segment_start + 1)
        result[segment_start:segment_start + segment_length - 1] = real(conv_segment[M:end])
        segment_start += L - M + 1
    end
    
    return result[1:N]
end

# problem 8.1.5
##
# funckja lti_filter
function lti_filter(b::Vector, a::Vector, x::Vector)::Vector
    N = length(x)
    K = length(a)
    M = length(b)
    y = zeros(Float64, N)

    for n in 1:N
        for m in 1:M
            if (n - m + 1) > 0
                y[n] += b[m] * x[n-m+1]
            end            
        end
        for k in 2:K
            if (n - k + 1) > 0
                y[n] -= a[k] * y[n-k+1]
            end
        end
        y[n] /= a[1]
    end
    return y
end

# problem 8.1.6
##
# funkcja filtfilt
function filtfilt(b::Vector, a::Vector, x::Vector)::Vector
    y_forward = lti_filter(b, a, x)
    y_reversed = reverse(y_forward)
    y_backward = lti_filter(b, a, y_reversed)
    y_filtfilt = reverse(y_backward)
    return y_filtfilt
end

# problem 8.1.7
##
#funkcja lti_amp
function lti_amp(f::Real, b::Vector, a::Vector)::Real
    M = length(b)
    K = length(a)
    ω = 2*π*f
    B = sum(b[m]*exp(-im*ω*(m-1)) for m in 1:M)
    A = sum(a[k]*exp(-im*ω*(k-1)) for k in 1:K)

    Ah = abs(B/A)
    return Ah
end

# problem 8.1.8
##
# funkcja lti_phase
function lti_phase(f::Real, b::Vector, a::Vector)::Real
    M = length(b)
    K = length(a)
    ω = 2*π*f
    B = sum(b[m]*exp(-im*ω*(m-1)) for m in 1:M)
    A = sum(a[k]*exp(-im*ω*(k-1)) for k in 1:K)
    ϕ = angle(B\A)
    return ϕ
end