# Zadania na dzisiaj
# 9.7, 10.1

# problem 9.7
##
# funkcja firwin_diff
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
# Przykładowe użycie
order = 21
fir_diff_filter = firwin_diff(order)
println(fir_diff_filter)

# problem 10.1
##
# funkcja resample
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

# Przykładowe użycie
x = [1.0, 2.0, 3.0, 4.0, 5.0]
M = 3
N = 2
K = 5
resampled_signal = resample(x, M, N, K)
println(resampled_signal)