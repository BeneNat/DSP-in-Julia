# Digital Signal Processing in Julia

This repository contains laboratory exercises and algorithm implementations for the Digital Signal Processing (Cyfrowe Przetwarzanie Sygnałów) course. The coursework was completed as part of the Electronics and Telecommunications curriculum at AGH University of Krakow.

## Project Description

The primary objective of this project is to implement, analyze, and visualize fundamental Digital Signal Processing (DSP) algorithms using the Julia programming language. The codebase covers the signal processing chain from signal generation and sampling to spectral analysis and digital filter design.

A custom module, `Modul_CPS`, is included to provide shared utility functions for signal generation and transformation, ensuring modularity and code reuse across different laboratory sessions.

## Repository Structure

The repository is organized into directory-based sessions, each addressing specific theoretical and practical aspects of DSP:

* **Lab01:** Introduction to Julia syntax, performance comparison between recursive and iterative algorithms, and floating-point arithmetic operations.
* **Lab02:** Modeling of continuous and discrete signals. Calculation of signal parameters including mean, energy, power, and RMS. Generation of standard test signals (unit impulse, ramp, sawtooth).
* **Lab03:** Analysis of the Sampling Theorem and aliasing effects. Signal reconstruction using interpolation methods. Quantization error analysis.
* **Lab04:** Implementation of the Discrete Fourier Transform (DFT) and Inverse DFT (IDFT). Computational complexity analysis.
* **Lab05:** Frequency domain analysis. Investigation of spectral leakage and Power Spectral Density (PSD) estimation.
* **Lab06:** Time-Frequency analysis. Implementation of the Short-Time Fourier Transform (STFT) and spectrogram visualization. Processing of audio signals (`.wav`).
* **Lab07:** Analysis of Linear Time-Invariant (LTI) systems. Implementation of convolution algorithms (direct, circular, overlap-add, overlap-save). Stability analysis.
* **Lab08:** Digital filter design using the window method. Analysis of Finite Impulse Response (FIR) filter characteristics.
* **Lab09:** Implementation of multirate signal processing techniques, including resampling, interpolation, and decimation.
* **Modul_CPS:** A reusable library containing core DSP functions and signal generators used throughout the coursework.

## Dependencies

The project utilizes the following Julia packages for computation and visualization:

* **CairoMakie / GLMakie**: High-performance plotting and visualization.
* **Plots**: Standard plotting interface.
* **DSP**: Digital Signal Processing utility library.
* **FFTW**: Fast Fourier Transform library (implicit dependency).
* **WAV**: Audio file I/O operations.
* **LinearAlgebra**: Matrix and vector operations.
* **Random**: Random number generation for noise simulation.

## Usage Instructions

To replicate the environment and execute the scripts, follow these steps:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/BeneNat/DSP-in-Julia
    cd DSP-in-Julia
    ```

2.  **Initialize the Julia Environment**
    Launch the Julia REPL in the project directory. Enter the package manager mode by pressing `]` and execute the following commands to install required dependencies:
    ```julia
    pkg> activate .
    pkg> instantiate
    ```

3.  **Execute Laboratory Scripts**
    Exit the package manager (press `Backspace`) and include the desired script. For example, to run Lab 01:
    ```julia
    include("Lab01/lab01.jl")
    ```
    *Note: Ensure the `Modul_CPS` is accessible in the load path if the specific script requires it.*

## Author and Context

* **Author:** Filip Żurek
* **Institution:** AGH University of Krakow
* **Faculty:** Faculty of Computer Science, Electronics and Telecommunications
* **Field of Study:** Electronics and Telecommunications
* **Course:** Digital Signal Processing (Cyfrowe Przetwarzanie Sygnałów)

## License

This software is distributed under the MIT License. Refer to the [LICENSE](LICENSE) file for the full text.

---
AGH University of Krakow - Digital Signal Processing Course 2024