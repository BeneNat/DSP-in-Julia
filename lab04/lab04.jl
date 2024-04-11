# zadania na dzisiaj
# 6.1, 6.2, 6.4, 6.5, 6.6, 6.7 -> do standardowego kolokwium zaliczeniowego
# 6.x -> rozne moga sie przydac i ulatwic do egzaminu z przedmiotu

# problem 6.1
##
# 2-DFT
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


# problem 6.5
##
# domena czestotliwosci -> domena czasu
# idft


# problem 6.6
##
# domena czasu -> domena czestotliwosci inny
# rdft


# problem 6.7
##
# domena czestotliwosci -> domena czasu inny
# irdft