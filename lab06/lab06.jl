# zadania na dzisiaj
# 7.2.8 - 7.2.11

# problem 7.2.8
##
# stft
function stft(x::Vector{Float64}, w::Vector{Float64}, L::Int)
    N = length(x)
    K = length(w)
    M = div(N-K, L) + 1

    stft_result = zeros(Complex{Float64}, M, div(K, 2) + 1)
    for m in 1:M
        # Indeksy próbek wycinka
        idx_start = 1 + (m - 1) * L
        idx_end = min(N, idx_start + K - 1)

        # Wyciągnięcie wycinka sygnału
        x_slice = x[idx_start:idx_end]

        # Nałożenie okna analizującego
        x_windowed = x_slice .* w

        # Obliczenie STFT za pomocą rfft
        stft_result[m, :] = rfft(x_windowed)
    end
    return stft_result
end

# problem 7.2.9
##
# analiza czas-czestotliwosciowa i spektrogramy podanych sygnalow
#= epic_sax_gux.wav,
fail_trombone.wav,
speech0001.wav. =#
using DSP
using Plots

# Funkcja do wyświetlania spektrogramu
function plot_spectrogram(filename, window_length, overlap)
    # Wczytanie sygnału audio
    audio, sample_rate = wavread(filename)

    # Obliczenie spektrogramu
    spec = spectrogram(audio, window_length, overlap)

    # Wyświetlenie spektrogramu
    heatmap(spec.freqs, spec.times, 10*log10.(spec.power), color = :turbo,
            xlabel = "Czas [s]", ylabel = "Częstotliwość [Hz]", title = "Spektrogram")
end

# Parametry analizy czasowo-częstotliwościowej
window_length = 512
overlap = 256

# Wyświetlenie spektrogramów dla każdego sygnału
plot_spectrogram("/Users/filipzurek/Documents/Studia/Elektronika i Telekomunikacja/Semestr_04/Cyfrowe przetwarzanie sygnalow/Laboratorium/Lab06/epic_sax_guy.wav", window_length, overlap)
plot_spectrogram("/Users/filipzurek/Documents/Studia/Elektronika i Telekomunikacja/Semestr_04/Cyfrowe przetwarzanie sygnalow/Laboratorium/Lab06/fail_trombone.wav", window_length, overlap)
plot_spectrogram("/Users/filipzurek/Documents/Studia/Elektronika i Telekomunikacja/Semestr_04/Cyfrowe przetwarzanie sygnalow/Laboratorium/Lab06/speech0001.wav", window_length, overlap)

# problem 7.2.10
##
# istft
using DSP

function istft(spec::Matrix{Complex{T}}, window::Vector{T}, overlap::Int) where T<:Real
    (num_freqs, num_frames) = size(spec)
    frame_length = length(window)
    signal_length = (num_frames - 1) * overlap + frame_length
    
    # Inicjalizacja sygnału wynikowego
    signal = zeros(T, signal_length)
    
    # Synteza sygnału
    for i in 1:num_frames
        idx_start = (i - 1) * overlap + 1
        idx_end = idx_start + frame_length - 1
        
        # Odwrotna DFT dla i-tego ramki
        frame = irdft(spec[:, i])
        
        # Nałożenie na sygnał ramki z uwzględnieniem nachodzenia
        signal[idx_start:idx_end] += frame .* window
    end
    
    return signal
end

#
##
# problem 7.2.11
##
# reprezentacja czasowo-czestotliwosciowa podanych sygnalow
#= epic_sax_gux.wav,
fail_trombone.wav,
speech0001.wav. =#
using DSP
using Plots

# Funkcja do obliczania i wyświetlania spektrogramu
function plot_spectrogram(filename)
    # Wczytanie sygnału audio
    audio, sample_rate = wavread(filename)

    # Parametry spektrogramu
    window_length = 512
    overlap = 256

    # Obliczenie spektrogramu
    spec = spectrogram(audio, window_length, overlap)

    # Wyświetlenie spektrogramu
    heatmap(spec.freqs, spec.times, 10*log10.(spec.power), color = :turbo,
            xlabel = "Czas [s]", ylabel = "Częstotliwość [Hz]", title = "Spektrogram")
end

# Wczytanie i wyświetlenie spektrogramów dla każdego sygnału
plot_spectrogram("/Users/filipzurek/Documents/Studia/Elektronika i Telekomunikacja/Semestr_04/Cyfrowe przetwarzanie sygnalow/Laboratorium/Lab06/epic_sax_guy.wav")
plot_spectrogram("/Users/filipzurek/Documents/Studia/Elektronika i Telekomunikacja/Semestr_04/Cyfrowe przetwarzanie sygnalow/Laboratorium/Lab06/fail_trombone.wav")
plot_spectrogram("/Users/filipzurek/Documents/Studia/Elektronika i Telekomunikacja/Semestr_04/Cyfrowe przetwarzanie sygnalow/Laboratorium/Lab06/speech0001.wav")


##
#test
using WAV
using DSP

# Wczytaj sygnały
epic_sax, fs = wavread("/Users/filipzurek/Documents/Studia/Elektronika i Telekomunikacja/Semestr_04/Cyfrowe przetwarzanie sygnalow/Laboratorium/Lab06/epic_sax_guy.wav")
fail_trombone, _ = wavread("/Users/filipzurek/Documents/Studia/Elektronika i Telekomunikacja/Semestr_04/Cyfrowe przetwarzanie sygnalow/Laboratorium/Lab06/fail_trombone.wav")
speech, _ = wavread("/Users/filipzurek/Documents/Studia/Elektronika i Telekomunikacja/Semestr_04/Cyfrowe przetwarzanie sygnalow/Laboratorium/Lab06/speech0001.wav")

# Parametry analizy
window_lengths = [256, 512, 1024]
overlap_ratios = [0.5, 0.75, 0.9]

for win_len in window_lengths
    for overlap_ratio in overlap_ratios
        # Wyświetl spektrogram dla epic_sax
        epic_sax_spec = spectrogram(epic_sax, win_len, win_len * overlap_ratio)
        heatmap(epic_sax_spec.t, epic_sax_spec.f, abs2.(epic_sax_spec.S), yflip=true)
        title!("Spektrogram epic_sax (Okno: $win_len, Overlap: $overlap_ratio)")
        xlabel!("Czas [s]")
        ylabel!("Częstotliwość [Hz]")
        savefig("epic_sax_spec_$(win_len)_$(overlap_ratio).png")

        # Wyświetl spektrogram dla fail_trombone
        fail_trombone_spec = spectrogram(fail_trombone, win_len, win_len * overlap_ratio)
        heatmap(fail_trombone_spec.t, fail_trombone_spec.f, abs2.(fail_trombone_spec.S), yflip=true)
        title!("Spektrogram fail_trombone (Okno: $win_len, Overlap: $overlap_ratio)")
        xlabel!("Czas [s]")
        ylabel!("Częstotliwość [Hz]")
        savefig("fail_trombone_spec_$(win_len)_$(overlap_ratio).png")

        # Wyświetl spektrogram dla speech
        speech_spec = spectrogram(speech, win_len, win_len * overlap_ratio)
        heatmap(speech_spec.t, speech_spec.f, abs2.(speech_spec.S), yflip=true)
        title!("Spektrogram speech (Okno: $win_len, Overlap: $overlap_ratio)")
        xlabel!("Czas [s]")
        ylabel!("Częstotliwość [Hz]")
        savefig("speech_spec_$(win_len)_$(overlap_ratio).png")
    end
end