module CPS

using LinearAlgebra

author = Dict{Symbol, String}(
    :index => "414487",
    :name  => "Filip Żurek",
    :email => "filipzurek@student.agh.edu.pl",
    :group => "7",
)

# Sygnaly ciagle
cw_rectangular(t::Real; T=1.0)::Real = abs(t) < T / 2 ? 1 : (abs(t) == T / 2 ? 0.5 : 0)
cw_triangle(t::Real; T=1.0)::Real = abs(t) < T ? 1 - abs(t) : 0
cw_literka_M(t::Real; T=1.0)::Real = abs(t) < T ? (t < 0 ? -t + 1 : t + 1) : 0
cw_literka_U(t::Real; T=1.0)::Real = abs(t) < T ? t^2 : 0

ramp_wave(t::Real)::Real = 2 * rem(t, 1, RoundNearest)
sawtooth_wave(t::Real)::Real = -2 * rem(t, 1, RoundNearest)
#triangular_wave(t::Real)::Real = ifelse(mod(t + 1 / 4, 1.0) < 1 / 2, 4*mod(t + 1 / 4, 1.0) - 1, -4*mod(t + 1 / 4, 1.0) + 3)
triangular_wave(t::Real)::Real = mod(t + 1 / 4, 1.0) < 1/2 ? 4*mod((t+1) / 4, 1.0) - 1 : -4*mod((t+1) / 4, 1.0) + 3
#square_wave(t::Real)::Real =  ifelse(mod(t, 1) < 0.5, 1, -1)
square_wave(t::Real)::Real = mod(t, 1) < 0.5 ? 1 : -1
#pulse_wave(t::Real, ρ::Real=0.2)::Real = ifelse(mod(t, 1) < ρ, 1, 0)
pulse_wave(t::Real, ρ::Real=0.2)::Real = mod(t, 1) < ρ ? 1 : 0
impuse_repeater(g::Function, t1::Real, t2::Real)::Function = x -> g(mod(x - t1, t2 - t1) + t1)

function ramp_wave_bl(t; A=1.0, T=1.0, band=20.0)
    signal = 0
    temp = 0
    n = 1
    while (arg = 2π * n * (1 / T)) < band * 2π
        temp += (-1)^n * sin.(arg * t) / n
        n += 1
    end
    signal += -2A / π * temp
    return signal
end

function sawtooth_wave_bl(t; A=1.0, T=1.0, band=20.0)
    signal = 0
    n = 1
    while (arg = 2π * n * (1 / T)) < band * 2π
        signal += (-1)^n * sin.(arg * t) / n
        n += 1
    end
    signal *= 2A / π
    return signal
end

function triangular_wave_bl(t; A=1.0, T=1.0, band=20.0)
    signal = 0
    n = 1
    while (arg = 2π * n * (1 / T)) < band * 2π
        signal += (-1)^((n - 1) / 2) * sin.(arg * t) / n^2
        n += 2
    end
    signal *= (8A / π^2)
    return signal
end

function square_wave_bl(t; A=1.0, T=1.0, band=20.0)
    signal = 0
    n = 1
    while (arg = 2π * (2n - 1) * (1 / T)) < band * 2π
        signal += sin.(arg * t) / (2n - 1)
        n += 1
    end
    signal *= 4 * A / π
    return signal
end

function pulse_wave_bl(t; ρ=0.2, A=1.0, T=1.0, band=20.0)
    signal = (sawtooth_wave_bl.(t .- (T / 2); A, T, band) - sawtooth_wave_bl.(t .- ((T / 2) + ρ); A, T, band)) .+ (2 * A * ρ)
    return signal
end


function impuse_repeater_bl(g::Function, t0::Real, t1::Real, band::Real)::Function
    T::Float64 = t2 - t1
    ω₀::Float64 = (2π / T)
    n_terms::Integer = div(band * 2π, ω₀)

    a0 = 1 / T * quadgk(g, t1, t2)[1]
    an_coeffs = zeros(Float64, n_terms)
    bn_coeffs = zeros(Float64, n_terms)

    for n in 1:n_terms
        an_coeffs[n] = 2 / T * quadgk(t -> g(t) * cos(ω₀ * n * t), t1, t2)[1]
        bn_coeffs[n] = 2 / T * quadgk(t -> g(t) * sin(ω₀ * n * t), t1, t2)[1]
    end

    function fourier_series_output(t::Float64)
        result = a0 / 2
        for n in 1:n_terms
            result += an_coeffs[n] * cos(ω₀ * n * t) + bn_coeffs[n] * sin(ω₀ * n * t)
        end
        return result
    end

    return fourier_series_output
end

function rand_siganl_bl(f1::Real, f2::Real)::Function
    f = f1 .+ rand(1000) .* (f2 - f1)
    ϕ = -π .+ rand(1000) * 2π
    A = randn(1000)
    A = A ./ sqrt(0.5 * sum(A .^ 2))
    return t -> sum(A .* sin.(2π * f .* t .+ ϕ))
end


# Sygnaly dyskretne
kronecker(n::Integer)::Real = n == 0 ? 1 : 0
heaviside(n::Integer)::Real = n < 0 ? 0 : 1

# Okna
rect(N::Integer)::AbstractVector{<:Real} = ones(N)
triang(N::Integer)::AbstractVector{<:Real} = [1 - (2*abs(n - ((N-1)/2))) / (N-1) for n = 0:N-1]
hanning(N::Integer)::AbstractVector{<:Real} = [0.5*(1 - cos(2*π*n/(N-1))) for n = 0:N-1]
hamming(N::Integer)::AbstractVector{<:Real} = [0.538 - 0.462*cos(2*π*n/(N-1)) for n = 0:N-1]
blackman(N::Integer)::AbstractVector{<:Real} = [0.42 - 0.5*cos(2*π*n/(N-1)) + 0.08*cos(4*π*n/(N-1)) for n = 0:N-1]

# Parametry sygnalow
mean(x::AbstractVector)::Number = sum(x)/length(x)
peak2peak(x::AbstractVector)::Real = abs(max(x) - min(x))
energy(x::AbstractVector)::Real = sum(x.*x)
power(x::AbstractVector)::Real = sum(x.*x)/length(x)
rms(x::AbstractVector)::Real = sqrt(sum(x.*x)/length(x))

function running_mean(x::AbstractVector, M::Integer)::Vector
    result::AbstractVector = zeros(length(x))
    for k in 1:length(x)
        n1 = k - M < 1 ? 1 : k-M
        n2 = k + M > lastindex(x) ? lastindex(x) : k + M
        result[k] = (1 / (n2-n1 + 1)) * sum(x[n1:n2])
    end
    return result
end

function running_energy(x::AbstractVector, M::Integer)::Vector
    result::AbstractVector = zeros(length(x))
    for k in 1:length(x)
        n1 = k - M < 1 ? 1 : k - M
        n2 = k + M > lastindex(x) ? lastindex(x) : k + M
        result[k] = sum(abs2, x[n1:n2])
    end
    return result
end

function running_power(x::AbstractVector, M::Integer)::Vector
    result::AbstractVector = zeros(length(x))
    for k in 1:length(x)
        n1 = k - M < 1 ? 1 : k - M
        n2 = k + M > lastindex(x) ? lastindex(x) : k + M
        result[k] = (1/(n2-n1 + 1)) * sum(abs2, x[n1:n2])
    end
    return result
end


# Probkowanie
function interpolate(
    m::AbstractVector,
    s::AbstractVector,
    kernel::Function=sinc
)::Real
    return x -> begin 
        sum = 0.0
        Δt = m[2] - m[1]
        for i in eachindex(m)
            sum += s[i] * kernel((x-m[i])/Δt)
        end
        return sum
    end
end

# Kwantyzacja
quantize(L::AbstractVector)::Function = x -> L[argmin(abs.(-L .+ x))]
SQNR(N::Integer)::Real = 6.02*N + 1.76
SNR(Psignal, Pnoise)::Real = 10*log10(Psignal/Pnoise)


# Obliczanie DFT
function dtft(f::Real; signal::AbstractVector, fs::Real)
    result::ComplexF64 = 0.0
    for n in eachindex(signal)
        result += signal[n] * cispi(-2*f*n / fs)
    end
    return result
 end

function dft(x::AbstractVector)::Vector
    N = length(x)
    X = zeros(ComplexF64, N)
    for k in 1:N
        for n in 1:N
            X[k] += x[n] * exp(-2im * π * (k-1) * (n-1) / N)
        end
    end
    return X
end

function idft(X::AbstractVector)::Vector
    N = length(x)
    X = zeros(ComplexF64, N)
    for n in 1:N
        for k in 1:N
            X[n] += x[k] * exp(2im * π * (k-1) * (n-1) / N)
        end
    end
    return X / N
end

function rdft(x::AbstractVector)::Vector
    N = length(x)
    X = Array{ComplexF64, 1}(undef, div(N, 2)+1)
    for k in 1:div(N, 2)+1
        X[k] = 0
        for n in 1:N
            X[k] += x[n] * exp(-2im * π * (k-1) * (n-1) / N)
        end
    end
    return X
end

function irdft(X::AbstractVector, N::Integer)::Vector
    N = (length(X)-1)*2
    x = Array{Float64, 1}(undef, N)
    for n in 1:N
        x[n] = real(X[1])
        for k in 2:eachindex(X)
            x[n] += 2*real(X[k] * exp(2im * π * (k-1) * (n-1)/ N))
        end
        x[n] /= N
    end
    return x
end

function fft_radix2_dit_r(x::AbstractVector)::Vector
    N = length(x)
    if N <= 1
        return x
    end
    if N % 2 != 0
        throw(ArgumentError("Długość sygnału musi być potęgą 2"))
    end
    even = fft_radix2_dit_r(x[1:2:end])
    odd = fft_radix2_dit_r(x[2:2:end])
    T = [exp(-2im * π * k / N) * odd[k] for k in 1:(N ÷ 2)]

    return vcat([even[k] + T[k] for k in 1:(N ÷ 2)], [even[k] - T[k] for k in 1:(N ÷ 2)])
end

function ifft_radix2_dif_r(X::AbstractVector)::Vector
    N = length(X)
    if N <= 1
        return X
    end
    if N % 2 != 0
        throw(ArgumentError("Długość sygnału musi być potęgą 2"))
    end

    X_even = X[1:(N ÷ 2)]
    X_odd = X[(N ÷ 2 + 1):end]
    T = [exp(2im * π * k / N) * X_odd[k] for k in 1:(N ÷ 2)]
    X_top = [X_even[k] + T[k] for k in 1:(N ÷ 2)]
    X_bottom = [X_even[k] - T[k] for k in 1:(N ÷ 2)]
    x_combined = vcat(X_top, X_bottom)

    return (1/N) * ifft_radix2_dif_r(x_combined)
end

function fft(x::Vector{Complex{Float64}})::Vector{Complex{Float64}}
    N = length(x)
    if N <= 1
        return x
    end
    even = fft(x[1:2:end])
    odd = fft(x[2:2:end])
    # Obliczamy współczynniki twiddle
    T = [exp(-2im * π * k / N) * odd[k] for k in 1:(div(N, 2))]
    # Łączymy wyniki
    return vcat([even[k] + T[k] for k in 1:(div(N, 2))], [even[k] - T[k] for k in 1:(div(N, 2))])
end

function ifft(X::AbstractVector)::Vector
    N = length(X)
    if N <= 1
        return X
    end
    X_conj = conj.(X)
    x_temp = fft(X_conj)
    return conj.(x_temp) / N
end

fftfreq(N::Integer, fs::Real)::Vector = [n * N / fs for n in 0:(N-1)]
rfftfreq(N::Integer, fs::Real)::Vector = [n * N / fs for n in 0:(N÷2)]
# fftfreq(N::Integer, fs::Real)::Vector = (0:N-1)*fp / N
# rfftfreq(N::Integer, fs::Real)::Vector = (0:floor(Int, N/2))*fp / N
amplitude_spectrum(x::AbstractVector, w::AbstractVector=rect(length(x)))::Vector = abs.(fft(x.*w))
power_spectrum(x::AbstractVector, w::AbstractVector=rect(length(x)))::Vector = abs2(fft(x.*w))
psd(x::AbstractVector, w::AbstractVector=rect(length(x)), fs::Real=1.0)::Vector = power_spectrum(x, w)/fs


function periodogram(x::Vector{Complex{Float64}}, fs::Float64, w::Vector{Float64}, L::Int)::Vector{Float64}
    N = length(x)
    K = length(w)
    S = div(N - K, L) + 1  # liczba segmentów
    Pxx = zeros(Float64, K)
    
    for i in 1:S
        start_idx = 1 + (i - 1) * L
        end_idx = start_idx + K - 1
        if end_idx > N
            break
        end
        segment = x[start_idx:end_idx]
        windowed_segment = segment .* w
        segment_fft = fft(windowed_segment)
        segment_psd = (1 / (fs * K)) * abs2.(segment_fft)
        Pxx += segment_psd
    end
    
    Pxx /= S
    return Pxx
end

function rfft(x::Vector{Float64})::Vector{Complex{Float64}}
    N = length(x)
    X = fft(complex(x))
    return X[1:(div(N, 2) + 1)]
end

function irfft(X::Vector{Complex{Float64}})::Vector{Float64}
    N = length(X)
    if N <= 1
        return real(X)
    end
    full_X = vcat(X, conj(reverse(X[2:end-1])))
    x_temp = fft(full_X)
    return real(x_temp) / length(full_X)
end

function stft(x::Vector{Float64}, w::Vector{Float64}, L::Int)::Matrix{Complex{Float64}}
    N = length(x)
    K = length(w)
    num_segments = div(N - K, L) + 1  # liczba segmentów
    stft_result = Matrix{Complex{Float64}}(undef, div(K, 2) + 1, num_segments)
    
    for i in 1:num_segments
        start_idx = 1 + (i - 1) * L
        end_idx = start_idx + K - 1
        if end_idx > N
            break
        end
        segment = x[start_idx:end_idx] .* w
        segment_fft = rfft(segment)
        stft_result[:, i] = segment_fft
    end
    
    return stft_result
end

function istft(X::Matrix{Complex{Float64}}, w::Vector{Float64}, L::Int)::Vector{Float64}
    K = length(w)
    num_segments = size(X, 2)
    N = K + (num_segments - 1) * L  # długość oryginalnego sygnału
    x = zeros(Float64, N)
    wsum = zeros(Float64, N)
    
    for i in 1:num_segments
        start_idx = 1 + (i - 1) * L
        end_idx = start_idx + K - 1
        segment = irfft(X[:, i]) .* w
        x[start_idx:end_idx] += segment
        wsum[start_idx:end_idx] += w
    end
    
    # Unikamy dzielenia przez zero
    wsum[wsum .== 0.0] .= 1.0
    return x ./ wsum
end

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

function fast_conv(f::Vector, g::Vector)::Vector
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

function filtfilt(b::Vector, a::Vector, x::Vector)::Vector
    y_forward = lti_filter(b, a, x)
    y_reversed = reverse(y_forward)
    y_backward = lti_filter(b, a, y_reversed)
    y_filtfilt = reverse(y_backward)
    return y_filtfilt
end

function lti_amp(f::Real, b::Vector, a::Vector)::Real
    M = length(b)
    K = length(a)
    ω = 2*π*f
    B = sum(b[m]*exp(-im*ω*(m-1)) for m in 1:M)
    A = sum(a[k]*exp(-im*ω*(k-1)) for k in 1:K)

    Ah = abs(B/A)
    return Ah
end

function lti_phase(f::Real, b::Vector, a::Vector)::Real
    M = length(b)
    K = length(a)
    ω = 2*π*f
    B = sum(b[m]*exp(-im*ω*(m-1)) for m in 1:M)
    A = sum(a[k]*exp(-im*ω*(k-1)) for k in 1:K)
    ϕ = angle(B\A)
    return ϕ
end


function firwin_lp_I(order, F0)
    L = div(order, 2)
    h = zeros(Float64, order + 1)
    
    for m in 0:order
        if m == L
            h[m + 1] = 2 * F0
        else
            h[m + 1] = sin(2 * π * F0 * (m - L)) / (π * (m - L))
        end
    end
    
    return h
end

function firwin_hp_I(order, F0)
    L = div(order, 2)
    h = zeros(Float64, order + 1)
    
    for m in 0:order
        if m == L
            h[m + 1] = 1 - 2 * F0
        else
            h[m + 1] = -sin(2 * π * F0 * (m - L)) / (π * (m - L))
        end
    end
    
    return h
end

function firwin_bp_I(order, F1, F2)
    h_lp_f2 = firwin_lp_I(order, F2)
    # h_lp_f1 = firwin_lp_I(order, F1)
    h_hp_f1 = firwin_hp_I(order, F1)

    h_bp = zeros(Float64, order + 1)

    for n in 1:(order+1)
        h_bp[n] = h_lp_f2[n] - h_hp_f1[n]
    end

    return h_bp
end

function firwin_bs_I(order, F1, F2)
    h_lp_f1 = firwin_lp_I(order, F1)
    h_hp_f2 = firwin_hp_I(order, F2)

    h_bs = zeros(Float64, order + 1)

    for n in 1:(order+1)
        h_bs[n] = h_lp_f1[n] + h_hp_f2[n]
    end

    return h_bs
end

function firwin_lp_II(order, F0)
    L = div(order, 2)
    h = zeros(Float64, order)

    for m in 1:order-1
        if m == L
            h[m+1] = 2 * F0
        else
            h[m+1] = sin(2*π*F0*(m-L)) / (π*(m-L))
        end
    end

    return h
end

function firwin_hp_II(order, F0)
    h_lp = firwin_lp_II(order, F0)
    N = order
    h_hp = zeros(Float64, order)

    for n in 1:order
        if n == div(order, 2) + 1
            h_hp[n] = 1.0 - h_lp[n]
        else
            h_hp[n] = -h_lp[n]
        end
    end
    
    return h_hp
end

function firwin_bp_II(order, F1, F2)
    h_lp_f2 = firwin_lp_II(order, F2)
    h_hp_f1 = firwin_hp_II(order, F1)

    h_bp = zeros(Float64, order)

    for n in 1:order
        h_bp[n] = h_lp_f2[n] - h_hp_f1[n]
    end

    return h_bp
end

function firwin_diff(N::Int)
    L = div(N, 2)
    h = zeros(Float64, N)

    for m in -L:L
        if m == 0
            h[L+m+1] = 0.0
        else
            h[L+m+1] = cos(π*m) / m
        end
    end

    return h
end

function linear_interpolate(x::Vector{Float64}, idx::Float64)::Float64
    i = floor(Int, idx)
    frac = idx - i
    if i < 1
        return x[1]
    elseif i >= length(x)
        return x[end]
    else
        return (1 - frac) * x[i] + frac * x[i + 1]
    end
end

function resample(x::Vector{Float64}, M::Int, N::Int, K::Int)::Vector{Float64}
    scaling_factor = N / M

    g = Vector{Float64}(undef, K)
    for n in 1:K
        idx = n * scaling_factor
        g[n] = linear_interpolate(x, idx)
    end
    
    return g
end


end