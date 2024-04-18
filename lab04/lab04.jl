# zadania na dzisiaj
# 6.1, 6.2, 6.4, 6.5, 6.6, 6.7 -> do standardowego kolokwium zaliczeniowego
# 6.x -> rozne moga sie przydac i ulatwic do egzaminu z przedmiotu

# problem 6.1
##
# 2-DFT
# do sprawdzenia
x = [20, 5]
N = length(x)
A = zeros(N)
wn = exp(im * (2 * π/N))
for k in 1:length(A)
    for n in 1:N
        A[k] = x[n] * wn^(-k * n)
    end
end
@show A

# problem 6.2
##
# 4-DFT
# do sprawdzenia
x = [3, 2, 5, 1]
N = length(x)
A = zeros(N)
wn = exp(im * (2 * π/N))
for k in 1:length(A)
    for n in 1:N
        A[k] = x[n] * wn^(-k * n)
    end
end
@show A

# problem 6.4
##
# domena czasu -> domena czestotliwosci
# dft
function dft(x)
    N = length(x)
    X = zeros(ComplexF64, N)
    for k in 1:N
        for n in 1:N
            X[k] += x[n] * exp(-2im * π * (k-1) * (n-1) / N)
        end
    end
    return X
end

X = [1+1im, 2+2im, 3+3im]
@show x = dft(X)

# problem 6.5
##
# domena czestotliwosci -> domena czasu
# idft
function idft(x)
    N = length(x)
    X = zeros(ComplexF64, N)
    for n in 1:N
        for k in 1:N
            X[n] += x[k] * exp(2im * π * (k-1) * (n-1) / N)
        end
    end
    return X / N
end

X = [1+1im, 2+2im, 3+3im]
@show x = idft(X)

# problem 6.6
##
# domena czasu -> domena czestotliwosci inny
# rdft
function rdft(x)
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

X = [1+1im, 2+2im, 3+3im]
@show x = rdft(X)

# problem 6.7
##
# domena czestotliwosci -> domena czasu inny
# irdft
function irdft(X)
    N = (length(X)-1)*2
    x = Array{Float64, 1}(undef, N)
    for n in 1:N
        x[n] = real(X[1])
        for k in 2:length(X)
            x[n] += 2*real(X[k] * exp(2im * π * (k-1) * (n-1)/ N))
        end
        x[n] /= N
    end
    return x
end

X = [1+1im, 2+2im, 3+3im]
@show x = irdft(X)