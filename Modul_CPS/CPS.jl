module CPS

using LinearAlgebra
using QuadGK
using OffsetArrays

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
triangular_wave(t::Real)::Real = ifelse(mod(t + 1 / 4, 1.0) < 1 / 2, 4mod(t + 1 / 4, 1.0) - 1, -4mod(t + 1 / 4, 1.0) + 3)
square_wave(t::Real)::Real =  ifelse(mod(t, 1) < 0.5, 1, -1)
pulse_wave(t::Real, ρ::Real=0.2)::Real = ifelse(mod(t, 1) < ρ, 1, 0)
impuse_repeater(g::Function, t1::Real, t2::Real)::Function = x -> g(mod(x - t1, t2 - t1) + t1)

function ramp_wave_bl(t; A=1.0, T=1.0, band=20.0)
    missing
end

function sawtooth_wave_bl(t; A=1.0, T=1.0, band=20.0)
    missing
end

function triangular_wave_bl(t; A=1.0, T=1.0, band=20.0)
    missing
end

function square_wave_bl(t; A=1.0, T=1.0, band=20.0)
    missing
end

function pulse_wave_bl(t; T=0.2, A=1.0, T=1.0, band=20.0)
    missing
end


function impuse_repeater_bl(g::Function, t0::Real, t1::Real, band::Real)::Function
end

function rand_siganl_bl(f1::Real, f2::Real)::Function
    missing
end


# Sygnaly dyskretne
kronecker(n::Integer)::Real = missing
heaviside(n::Integer)::Real = missing

# Okna
rect(N::Integer)::AbstractVector{<:Real} = missing
triang(N::Integer)::AbstractVector{<:Real} = missing
hanning(N::Integer)::AbstractVector{<:Real} = missing
hamming(N::Integer)::AbstractVector{<:Real} = missing
blackman(N::Integer)::AbstractVector{<:Real} = missing

# Parametry sygnalow
mean(x::AbstractVector)::Number = missing
peak2peak(x::AbstractVector)::Real = missing
energy(x::AbstractVector)::Real = missing
power(x::AbstractVector)::Real = missing
rms(x::AbstractVector)::Real = missing

function running_mean(x::AbstractVector, M::Integer)::Vector
    missing
end

function running_energy(x::AbstractVector, M::Integer)::Vector
    missing
end

function running_power(x::AbstractVector, M::Integer)::Vector
    missing
end



# Probkowanie
function interpolate(
    m::AbstractVector,
    s::AbstractVector,
    kernel::Function=sinc
)::Real
    missing
end

# Kwantyzacja
quantize(L::AbstractVector)::Function = missing
SQNR(N::Integer)::Real = missing
SNR(Psignal, Pnoise)::Real = missing


# Obliczanie DFT
function dtft(f::Real; signal::AbstractVector, fs::Real)
   missing
end

function dft(x::AbstractVector)::Vector
    missing
end

function idft(X::AbstractVector)::Vector
   missing
end

function rdft(x::AbstractVector)::Vector
   missing
end

function irdft(X::AbstractVector, N::Integer)::Vector
   missing
end

function fft_radix2_dit_r(x::AbstractVector)::Vector
   missing
end

function ifft_radix2_dif_r(X::AbstractVector)::Vector
   missing
end

function fft(x::AbstractVector)::Vector
    dft(x) # Moze da rade lepiej?
end

function ifft(X::AbstractVector)::Vector
    idft(X) # Moze da rade lepiej?
end


fftfreq(N::Integer, fs::Real)::Vector = missing
rfftfreq(N::Integer, fs::Real)::Vector = missing
amplitude_spectrum(x::AbstractVector, w::AbstractVector=rect(length(x)))::Vector = missing
power_spectrum(x::AbstractVector, w::AbstractVector=rect(length(x)))::Vector = missing
psd(x::AbstractVector, w::AbstractVector=rect(length(x)), fs::Real=1.0)::Vector = missing

function periodogram(x::AbstractVector, w::AbstractVector=rect(length(x)), fs::Real=1.0)::Vector
    missing
end



function stft(x::AbstractVector, w::AbstractVector, L::Integer)::Matrix
    missing
end


function istft(X::AbstractMatrix{<:Complex}, w::AbstractVector{<:Real}, L::Integer)::AbstractVector{<:Real}
    missing
end

function conv(f::Vector, g::Vector)::Vector
    missing
end

function fast_conv(f::Vector, g::Vector)::Vector
    missing
end

function overlap_add(x::Vector, h::Vector, L::Integer)::Vector
    missing
end

function overlap_save(x::Vector, h::Vector, L::Integer)::Vector
    missing
end

function lti_filter(b::Vector, a::Vector, x::Vector)::Vector
    missing
end


end
