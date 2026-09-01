#!/usr/bin/env python3
"""
Frequency-domain analysis: FFT, PSD, spectrogram and CWT.

This script is intentionally structured in parallel with the
time-domain analysis script.

INPUT STRUCTURE
---------------

    root/
        servo_only/
            servo_only_trial01.csv
            servo_only_trial02.csv
            ...

        piston_only/
            piston_only_trial01.csv
            piston_only_trial02.csv
            ...

        both_simultaneous/
            both_simultaneous_trial01.csv
            both_simultaneous_trial02.csv
            ...

        both_alternating/
            both_alternating_trial01.csv
            both_alternating_trial02.csv
            ...

        Stationary_benchmark.csv


OUTPUT STRUCTURE
----------------

    frequency_domain_analysis/
        servo_only/
            servo_only_trial01/
                servo_only_trial01_frequency_domain_metrics.csv
                servo_only_trial01_ax_fft.csv
                servo_only_trial01_ax_fft_spectrum.png
                servo_only_trial01_ax_psd.csv
                servo_only_trial01_ax_psd.png
                servo_only_trial01_ax_fft_peaks.csv
                servo_only_trial01_ax_spectrogram.png
                servo_only_trial01_ax_cwt.csv
                servo_only_trial01_ax_cwt_scalogram.png
                servo_only_trial01_ax_cwt_power.png
                servo_only_trial01_ax_cwt_global_spectrum.csv
                servo_only_trial01_ax_cwt_global_spectrum.png
                servo_only_trial01_ax_cwt_wavelet.csv
                servo_only_trial01_ax_cwt_wavelet.png
                ...
            
            servo_only_trial02/
                ...

        piston_only/
            piston_only_trial01/
                ...

        both_simultaneous/
            both_simultaneous_trial01/
                ...

        both_alternating/
            both_alternating_trial01/
                ...

        stationary/
            Stationary_benchmark/
                ...

        summary_servo_only.csv
        summary_piston_only.csv
        summary_both_simultaneous.csv
        summary_both_alternating.csv
        all_trials_combined.csv


SIGNALS
-------

    ax   = x-axis acceleration
    ay   = y-axis acceleration
    az   = z-axis acceleration

    aabs = acceleration magnitude

           sqrt(ax^2 + ay^2 + az^2)


FFT
---

    One-sided amplitude spectrum.

    Metrics:
        Dominant frequency
        Dominant amplitude
        Spectral centroid
        Spectral bandwidth
        Spectral RMS
        Total spectral power
        F95


PSD
---

    Welch power spectral density.

    Units:
        (m/s^2)^2 / Hz


SPECTROGRAM
-----------

    STFT-based time-frequency PSD representation.


CWT
---

    Continuous Wavelet Transform using a complex Morlet wavelet.

    Outputs:
        CWT coefficient amplitude
        CWT power
        CWT amplitude scalogram
        CWT power scalogram
        Global wavelet spectrum
        Representative Morlet wavelet waveform

    The CWT is implemented directly and does not depend on
    scipy.signal.cwt being available.


SAMPLING FREQUENCY
------------------

    FS_TARGET = 402.1 Hz

    If a valid time column exists, the actual sampling frequency
    is estimated from the median time difference.

    Otherwise FS_TARGET is used.


PREPROCESSING
-------------

    The mean/DC component is removed before FFT, PSD and CWT
    calculations.

    This prevents the DC component from dominating the frequency
    domain analysis.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import (
    welch,
    spectrogram,
    find_peaks
)


# ============================================================
# USER SETTINGS
# ============================================================

CLASS_DIRS = [
    "servo_only",
    "piston_only",
    "both_simultaneous",
    "both_alternating"
]

AXES = [
    "ax",
    "ay",
    "az",
    "aabs"
]

SKIP_KEYWORDS = [
    "label",
    "manifest"
]


# ============================================================
# SAMPLING
# ============================================================

FS_TARGET = 402.1


# ============================================================
# PSD SETTINGS
# ============================================================

PSD_NPERSEG = 1024
PSD_NOVERLAP = 512


# ============================================================
# SPECTROGRAM SETTINGS
# ============================================================

SPECTROGRAM_NPERSEG = 256
SPECTROGRAM_NOVERLAP = 192


# ============================================================
# CWT SETTINGS
# ============================================================

CWT_MIN_FREQ = 1.0

# None = automatically use 95% of Nyquist
CWT_MAX_FREQ = None

# Number of logarithmically spaced CWT frequencies
N_CWT_FREQS = 128

# Morlet wavelet parameter
MORLET_W = 6.0


# ============================================================
# FFT PEAK SETTINGS
# ============================================================

N_TOP_PEAKS = 10


# ============================================================
# PLOT SETTINGS
# ============================================================

FIG_DPI = 300

plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "font.size": 10
})


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    df = pd.read_csv(path)

    n_cols = df.shape[1]

    base_names = [
        "t",
        "ax",
        "ay",
        "az",
        "aabs"
    ]

    if n_cols <= 5:

        df.columns = (
            base_names[:n_cols]
            + list(df.columns[n_cols:])
        )

    if "aabs" not in df.columns:

        df["aabs"] = np.sqrt(
            df["ax"] ** 2
            + df["ay"] ** 2
            + df["az"] ** 2
        )

    return df


# ============================================================
# ESTIMATE SAMPLING FREQUENCY
# ============================================================

def estimate_fs(df):

    if "t" not in df.columns:

        return FS_TARGET

    t = np.asarray(
        df["t"],
        dtype=float
    )

    if len(t) < 2:

        return FS_TARGET

    valid = np.isfinite(t)

    t = t[valid]

    if len(t) < 2:

        return FS_TARGET

    dt = np.diff(t)

    dt = dt[
        np.isfinite(dt)
        & (dt > 0)
    ]

    if len(dt) == 0:

        return FS_TARGET

    median_dt = np.median(dt)

    if median_dt <= 0:

        return FS_TARGET

    fs = 1.0 / median_dt

    if not np.isfinite(fs):

        return FS_TARGET

    return float(fs)


# ============================================================
# PREPARE SIGNAL
# ============================================================

def prepare_signal(x):

    x = np.asarray(
        x,
        dtype=float
    )

    if len(x) == 0:

        return x

    # Replace invalid values by interpolation
    if np.any(~np.isfinite(x)):

        idx = np.arange(
            len(x)
        )

        valid = np.isfinite(x)

        if np.sum(valid) < 2:

            return np.array([])

        x = np.interp(
            idx,
            idx[valid],
            x[valid]
        )

    # Remove DC component
    x = (
        x
        - np.mean(x)
    )

    return x


# ============================================================
# FFT
# ============================================================

def compute_fft(x, fs):

    n = len(x)

    if n < 2:

        return (
            np.array([]),
            np.array([])
        )

    X = np.fft.rfft(x)

    f = np.fft.rfftfreq(
        n,
        d=1.0 / fs
    )

    amplitude = (
        np.abs(X)
        / n
    )

    # Convert to one-sided amplitude spectrum.
    #
    # DC is not doubled.
    #
    # Nyquist is not doubled when N is even.
    #
    # All other positive-frequency components are doubled.

    if n % 2 == 0:

        if len(amplitude) > 2:

            amplitude[1:-1] *= 2.0

    else:

        if len(amplitude) > 1:

            amplitude[1:] *= 2.0

    return (
        f,
        amplitude
    )


# ============================================================
# PSD
# ============================================================

def compute_psd(x, fs):

    if len(x) < 4:

        return (
            np.array([]),
            np.array([])
        )

    nperseg = min(
        PSD_NPERSEG,
        len(x)
    )

    noverlap = min(
        PSD_NOVERLAP,
        nperseg - 1
    )

    f, psd = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density"
    )

    return (
        f,
        psd
    )


# ============================================================
# FREQUENCY-DOMAIN METRICS
# ============================================================

def compute_frequency_metrics(
    f_fft,
    amplitude,
    f_psd,
    psd
):

    result = {

        "DominantFrequency_Hz":
            np.nan,

        "DominantAmplitude":
            np.nan,

        "SpectralCentroid_Hz":
            np.nan,

        "SpectralBandwidth_Hz":
            np.nan,

        "SpectralRMS":
            np.nan,

        "TotalSpectralPower":
            np.nan,

        "F95_Hz":
            np.nan
    }

    # --------------------------------------------------------
    # Dominant FFT frequency
    # --------------------------------------------------------

    if len(f_fft) > 1:

        mask = (
            f_fft > 0
        )

        if np.any(mask):

            f_pos = f_fft[
                mask
            ]

            a_pos = amplitude[
                mask
            ]

            idx = np.argmax(
                a_pos
            )

            result[
                "DominantFrequency_Hz"
            ] = float(
                f_pos[idx]
            )

            result[
                "DominantAmplitude"
            ] = float(
                a_pos[idx]
            )

    # --------------------------------------------------------
    # PSD
    # --------------------------------------------------------

    if len(f_psd) < 2:

        return result

    power = np.maximum(
        psd,
        0.0
    )

    # np.trapz is intentionally used instead of np.trapezoid
    # for compatibility with older NumPy versions.

    total_power = np.trapz(
        power,
        f_psd
    )

    if (
        not np.isfinite(
            total_power
        )
        or total_power <= 0
    ):

        return result

    # --------------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------------

    centroid = (
        np.trapz(
            f_psd * power,
            f_psd
        )
        / total_power
    )

    result[
        "SpectralCentroid_Hz"
    ] = float(
        centroid
    )

    # --------------------------------------------------------
    # Spectral bandwidth
    # --------------------------------------------------------

    variance = (
        np.trapz(
            (
                (f_psd - centroid) ** 2
            )
            * power,
            f_psd
        )
        / total_power
    )

    result[
        "SpectralBandwidth_Hz"
    ] = float(
        np.sqrt(
            max(
                variance,
                0.0
            )
        )
    )

    # --------------------------------------------------------
    # Total spectral power
    # --------------------------------------------------------

    result[
        "TotalSpectralPower"
    ] = float(
        total_power
    )

    # --------------------------------------------------------
    # Spectral RMS
    # --------------------------------------------------------

    result[
        "SpectralRMS"
    ] = float(
        np.sqrt(
            total_power
        )
    )

    # --------------------------------------------------------
    # F95
    # --------------------------------------------------------

    cumulative = np.cumsum(
        power
    )

    if cumulative[-1] > 0:

        cumulative = (
            cumulative
            / cumulative[-1]
        )

        idx95 = np.searchsorted(
            cumulative,
            0.95
        )

        idx95 = min(
            idx95,
            len(f_psd) - 1
        )

        result[
            "F95_Hz"
        ] = float(
            f_psd[idx95]
        )

    return result


# ============================================================
# FFT PEAKS
# ============================================================

def find_fft_peaks(
    f,
    amplitude,
    n_peaks=N_TOP_PEAKS
):

    if len(f) < 3:

        return pd.DataFrame()

    mask = (
        f > 0
    )

    f_pos = f[
        mask
    ]

    a_pos = amplitude[
        mask
    ]

    peaks, _ = find_peaks(
        a_pos
    )

    if len(peaks) == 0:

        return pd.DataFrame()

    peak_freqs = f_pos[
        peaks
    ]

    peak_amps = a_pos[
        peaks
    ]

    order = np.argsort(
        peak_amps
    )[::-1]

    order = order[
        :n_peaks
    ]

    return pd.DataFrame({

        "Rank":
            np.arange(
                1,
                len(order) + 1
            ),

        "Frequency_Hz":
            peak_freqs[
                order
            ],

        "Amplitude":
            peak_amps[
                order
            ]
    })


# ============================================================
# FFT SPECTRUM PLOT
# ============================================================

def plot_fft_spectrum(
    f,
    amplitude,
    axis,
    fs,
    output_path
):

    if len(f) == 0:

        return

    nyquist = (
        fs / 2.0
    )

    mask = (
        (f >= 0)
        & (f <= nyquist)
    )

    plt.figure(
        figsize=(8, 4.5)
    )

    plt.plot(
        f[mask],
        amplitude[mask],
        linewidth=1.0
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Amplitude (m/s²)"
    )

    plt.title(
        f"One-Sided FFT Amplitude Spectrum — {axis}"
    )

    plt.xlim(
        0,
        nyquist
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PSD PLOT
# ============================================================

def plot_psd(
    f,
    psd,
    axis,
    fs,
    output_path
):

    if len(f) == 0:

        return

    nyquist = (
        fs / 2.0
    )

    mask = (
        (f >= 0)
        & (f <= nyquist)
    )

    plt.figure(
        figsize=(8, 4.5)
    )

    positive_psd = np.maximum(
        psd[mask],
        np.finfo(float).tiny
    )

    plt.semilogy(
        f[mask],
        positive_psd,
        linewidth=1.0
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "PSD ((m/s²)²/Hz)"
    )

    plt.title(
        f"Power Spectral Density — {axis}"
    )

    plt.xlim(
        0,
        nyquist
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SPECTROGRAM
# ============================================================

def generate_spectrogram(
    x,
    fs,
    axis,
    output_path
):

    if len(x) < 16:

        return

    nperseg = min(
        SPECTROGRAM_NPERSEG,
        len(x)
    )

    noverlap = min(
        SPECTROGRAM_NOVERLAP,
        nperseg - 1
    )

    f, t, Sxx = spectrogram(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="psd"
    )

    nyquist = (
        fs / 2.0
    )

    mask = (
        f <= nyquist
    )

    Sxx_plot = np.maximum(
        Sxx[mask],
        np.finfo(float).tiny
    )

    plt.figure(
        figsize=(8, 4.8)
    )

    plt.pcolormesh(
        t,
        f[mask],
        10.0 * np.log10(
            Sxx_plot
        ),
        shading="auto"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.title(
        f"Spectrogram — {axis}"
    )

    plt.ylim(
        0,
        nyquist
    )

    plt.colorbar(
        label="PSD (dB/Hz)"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MORLET WAVELET
# ============================================================

def morlet_wavelet(
    length,
    scale,
    w=MORLET_W
):

    if length < 3:

        return np.array([])

    n = np.arange(
        length
    )

    center = (
        length - 1
    ) / 2.0

    tau = (
        n - center
    ) / scale

    wavelet = (
        np.pi ** (-0.25)
        * np.exp(
            1j * w * tau
        )
        * np.exp(
            -(tau ** 2) / 2.0
        )
    )

    wavelet = (
        wavelet
        / np.sqrt(scale)
    )

    return wavelet


# ============================================================
# CWT
# ============================================================

def compute_cwt(
    x,
    fs,
    frequencies
):

    n = len(x)

    if n < 16:

        return (
            np.empty(
                (0, 0),
                dtype=complex
            ),
            np.array([])
        )

    # Morlet scale-frequency relation:
    #
    # f = w * fs / (2*pi*scale)
    #
    # scale = w*fs/(2*pi*f)

    scales = (
        MORLET_W
        * fs
        / (
            2.0
            * np.pi
            * frequencies
        )
    )

    coefficients = np.zeros(
        (
            len(scales),
            n
        ),
        dtype=complex
    )

    for i, scale in enumerate(
        scales
    ):

        wavelet_length = int(
            np.ceil(
                10.0 * scale
            )
        )

        wavelet_length = max(
            wavelet_length,
            32
        )

        wavelet_length = min(
            wavelet_length,
            n
        )

        if wavelet_length % 2 == 0:

            wavelet_length -= 1

        if wavelet_length < 3:

            continue

        wavelet = morlet_wavelet(
            wavelet_length,
            scale,
            MORLET_W
        )

        coefficients[i, :] = np.convolve(
            x,
            np.conjugate(
                wavelet[::-1]
            ),
            mode="same"
        )

    return (
        coefficients,
        scales
    )


# ============================================================
# CWT METRICS
# ============================================================

def compute_cwt_metrics(
    coefficients,
    frequencies
):

    result = {

        "CWT_DominantFrequency_Hz":
            np.nan,

        "CWT_DominantAmplitude":
            np.nan,

        "CWT_GlobalWaveletPower":
            np.nan,

        "CWT_MeanWaveletPower":
            np.nan
    }

    if coefficients.size == 0:

        return result

    power = (
        np.abs(
            coefficients
        ) ** 2
    )

    # --------------------------------------------------------
    # Global wavelet spectrum
    # --------------------------------------------------------

    global_power = np.mean(
        power,
        axis=1
    )

    idx = np.argmax(
        global_power
    )

    result[
        "CWT_DominantFrequency_Hz"
    ] = float(
        frequencies[idx]
    )

    result[
        "CWT_DominantAmplitude"
    ] = float(
        np.sqrt(
            global_power[idx]
        )
    )

    # --------------------------------------------------------
    # Integrated global wavelet power
    # --------------------------------------------------------

    order = np.argsort(
        frequencies
    )

    f_sorted = frequencies[
        order
    ]

    gp_sorted = global_power[
        order
    ]

    result[
        "CWT_GlobalWaveletPower"
    ] = float(
        np.trapz(
            gp_sorted,
            f_sorted
        )
    )

    # --------------------------------------------------------
    # Mean wavelet power
    # --------------------------------------------------------

    result[
        "CWT_MeanWaveletPower"
    ] = float(
        np.mean(
            power
        )
    )

    return result


# ============================================================
# SAVE CWT DATA
# ============================================================

def save_cwt_csv(
    coefficients,
    frequencies,
    t,
    output_path
):

    if coefficients.size == 0:

        return

    amplitude = np.abs(
        coefficients
    )

    power = (
        amplitude ** 2
    )

    data = {
        "time_s":
            t
    }

    for i, freq in enumerate(
        frequencies
    ):

        data[
            f"amplitude_{freq:.4f}_Hz"
        ] = amplitude[
            i,
            :
        ]

        data[
            f"power_{freq:.4f}_Hz"
        ] = power[
            i,
            :
        ]

    pd.DataFrame(
        data
    ).to_csv(
        output_path,
        index=False
    )


# ============================================================
# CWT AMPLITUDE SCALOGRAM
# ============================================================

def plot_cwt_scalogram(
    coefficients,
    frequencies,
    t,
    axis,
    output_path
):

    if coefficients.size == 0:

        return

    amplitude = np.abs(
        coefficients
    )

    plt.figure(
        figsize=(8, 4.8)
    )

    plt.pcolormesh(
        t,
        frequencies,
        amplitude,
        shading="auto"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.title(
        f"CWT Amplitude Scalogram — {axis}"
    )

    plt.yscale(
        "log"
    )

    plt.colorbar(
        label="Wavelet Coefficient Amplitude"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CWT POWER SCALOGRAM
# ============================================================

def plot_cwt_power(
    coefficients,
    frequencies,
    t,
    axis,
    output_path
):

    if coefficients.size == 0:

        return

    power = (
        np.abs(
            coefficients
        ) ** 2
    )

    power = np.maximum(
        power,
        np.finfo(float).tiny
    )

    plt.figure(
        figsize=(8, 4.8)
    )

    plt.pcolormesh(
        t,
        frequencies,
        10.0 * np.log10(
            power
        ),
        shading="auto"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.title(
        f"CWT Wavelet Power — {axis}"
    )

    plt.yscale(
        "log"
    )

    plt.colorbar(
        label="Wavelet Power (dB)"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# GLOBAL WAVELET SPECTRUM
# ============================================================

def save_and_plot_global_wavelet(
    coefficients,
    frequencies,
    axis,
    csv_path,
    plot_path
):

    if coefficients.size == 0:

        return

    power = (
        np.abs(
            coefficients
        ) ** 2
    )

    global_power = np.mean(
        power,
        axis=1
    )

    global_df = pd.DataFrame({

        "Frequency_Hz":
            frequencies,

        "GlobalWaveletPower":
            global_power,

        "GlobalWaveletAmplitude":
            np.sqrt(
                global_power
            )
    })

    global_df.to_csv(
        csv_path,
        index=False
    )

    order = np.argsort(
        frequencies
    )

    f_plot = frequencies[
        order
    ]

    p_plot = global_power[
        order
    ]

    plt.figure(
        figsize=(8, 4.5)
    )

    plt.plot(
        f_plot,
        p_plot,
        linewidth=1.2
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Global Wavelet Power"
    )

    plt.title(
        f"Global Wavelet Spectrum — {axis}"
    )

    plt.xscale(
        "log"
    )

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# REPRESENTATIVE WAVELET
# ============================================================

def save_and_plot_wavelet(
    fs,
    scale,
    frequency,
    axis,
    csv_path,
    plot_path
):

    wavelet_length = int(
        np.ceil(
            10.0 * scale
        )
    )

    wavelet_length = max(
        wavelet_length,
        128
    )

    wavelet_length = min(
        wavelet_length,
        2047
    )

    if wavelet_length % 2 == 0:

        wavelet_length -= 1

    wavelet = morlet_wavelet(
        wavelet_length,
        scale,
        MORLET_W
    )

    center = (
        wavelet_length - 1
    ) / 2.0

    t_wavelet = (
        np.arange(
            wavelet_length
        )
        - center
    ) / fs

    real_part = np.real(
        wavelet
    )

    imaginary_part = np.imag(
        wavelet
    )

    magnitude = np.abs(
        wavelet
    )

    wavelet_df = pd.DataFrame({

        "Time_s":
            t_wavelet,

        "Real":
            real_part,

        "Imaginary":
            imaginary_part,

        "Magnitude":
            magnitude
    })

    wavelet_df.to_csv(
        csv_path,
        index=False
    )

    plt.figure(
        figsize=(8, 4.5)
    )

    plt.plot(
        t_wavelet,
        real_part,
        linewidth=1.2,
        label="Real part"
    )

    plt.plot(
        t_wavelet,
        imaginary_part,
        linewidth=1.0,
        linestyle="--",
        label="Imaginary part"
    )

    plt.plot(
        t_wavelet,
        magnitude,
        linewidth=1.0,
        linestyle=":",
        label="Magnitude"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        f"Representative Morlet Wavelet — {axis}\n"
        f"Centre frequency = {frequency:.2f} Hz"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ANALYZE ONE FILE
# ============================================================

def analyze_file(
    path,
    output_dir
):

    df = load_csv(
        path
    )

    fs = estimate_fs(
        df
    )

    nyquist = (
        fs / 2.0
    )

    # --------------------------------------------------------
    # CWT frequency range
    # --------------------------------------------------------

    cwt_min_freq = max(
        CWT_MIN_FREQ,
        0.1
    )

    if CWT_MAX_FREQ is None:

        cwt_max_freq = (
            0.95
            * nyquist
        )

    else:

        cwt_max_freq = min(
            CWT_MAX_FREQ,
            0.95 * nyquist
        )

    if cwt_max_freq <= cwt_min_freq:

        raise ValueError(
            "CWT maximum frequency must be "
            "greater than minimum frequency."
        )

    frequencies = np.geomspace(
        cwt_min_freq,
        cwt_max_freq,
        N_CWT_FREQS
    )

    # --------------------------------------------------------
    # Time vector
    # --------------------------------------------------------

    if "t" in df.columns:

        t = np.asarray(
            df["t"],
            dtype=float
        )

        if (
            len(t) != len(df)
            or np.any(
                ~np.isfinite(t)
            )
        ):

            t = (
                np.arange(
                    len(df)
                )
                / fs
            )

    else:

        t = (
            np.arange(
                len(df)
            )
            / fs
        )

    # --------------------------------------------------------
    # Input file name
    # --------------------------------------------------------

    fname = os.path.splitext(
        os.path.basename(path)
    )[0]

    metric_rows = []

    # ========================================================
    # PROCESS AXES
    # ========================================================

    for axis in AXES:

        print(
            f"    Processing {axis}..."
        )

        x = prepare_signal(
            df[axis].values
        )

        if len(x) < 4:

            print(
                f"      Skipping {axis}: "
                f"insufficient samples."
            )

            continue

        # ====================================================
        # FFT
        # ====================================================

        f_fft, amplitude = compute_fft(
            x,
            fs
        )

        fft_df = pd.DataFrame({

            "Frequency_Hz":
                f_fft,

            "Amplitude_m_s2":
                amplitude
        })

        fft_csv_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_fft.csv"
        )

        fft_df.to_csv(
            fft_csv_path,
            index=False
        )

        # ----------------------------------------------------
        # FFT plot
        # ----------------------------------------------------

        fft_plot_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_fft_spectrum.png"
        )

        plot_fft_spectrum(
            f_fft,
            amplitude,
            axis,
            fs,
            fft_plot_path
        )

        # ====================================================
        # PSD
        # ====================================================

        f_psd, psd = compute_psd(
            x,
            fs
        )

        psd_df = pd.DataFrame({

            "Frequency_Hz":
                f_psd,

            "PSD_(m_s2)^2_per_Hz":
                psd
        })

        psd_csv_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_psd.csv"
        )

        psd_df.to_csv(
            psd_csv_path,
            index=False
        )

        # ----------------------------------------------------
        # PSD plot
        # ----------------------------------------------------

        psd_plot_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_psd.png"
        )

        plot_psd(
            f_psd,
            psd,
            axis,
            fs,
            psd_plot_path
        )

        # ====================================================
        # FREQUENCY METRICS
        # ====================================================

        fft_metrics = (
            compute_frequency_metrics(
                f_fft,
                amplitude,
                f_psd,
                psd
            )
        )

        # ====================================================
        # FFT PEAKS
        # ====================================================

        peaks_df = find_fft_peaks(
            f_fft,
            amplitude,
            N_TOP_PEAKS
        )

        if not peaks_df.empty:

            peaks_path = os.path.join(
                output_dir,
                f"{fname}_{axis}_fft_peaks.csv"
            )

            peaks_df.to_csv(
                peaks_path,
                index=False
            )

        # ====================================================
        # SPECTROGRAM
        # ====================================================

        spectrogram_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_spectrogram.png"
        )

        generate_spectrogram(
            x,
            fs,
            axis,
            spectrogram_path
        )

        # ====================================================
        # CWT
        # ====================================================

        coefficients, scales = compute_cwt(
            x,
            fs,
            frequencies
        )

        cwt_metrics = (
            compute_cwt_metrics(
                coefficients,
                frequencies
            )
        )

        # ====================================================
        # CWT DATA
        # ====================================================

        cwt_csv_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_cwt.csv"
        )

        save_cwt_csv(
            coefficients,
            frequencies,
            t,
            cwt_csv_path
        )

        # ====================================================
        # CWT AMPLITUDE SCALOGRAM
        # ====================================================

        cwt_scalogram_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_cwt_scalogram.png"
        )

        plot_cwt_scalogram(
            coefficients,
            frequencies,
            t,
            axis,
            cwt_scalogram_path
        )

        # ====================================================
        # CWT POWER SCALOGRAM
        # ====================================================

        cwt_power_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_cwt_power.png"
        )

        plot_cwt_power(
            coefficients,
            frequencies,
            t,
            axis,
            cwt_power_path
        )

        # ====================================================
        # GLOBAL WAVELET SPECTRUM
        # ====================================================

        global_csv_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_cwt_global_spectrum.csv"
        )

        global_plot_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_cwt_global_spectrum.png"
        )

        save_and_plot_global_wavelet(
            coefficients,
            frequencies,
            axis,
            global_csv_path,
            global_plot_path
        )

        # ====================================================
        # REPRESENTATIVE WAVELET
        # ====================================================

        dominant_cwt_frequency = (
            cwt_metrics[
                "CWT_DominantFrequency_Hz"
            ]
        )

        if np.isfinite(
            dominant_cwt_frequency
        ):

            representative_scale = (
                MORLET_W
                * fs
                / (
                    2.0
                    * np.pi
                    * dominant_cwt_frequency
                )
            )

        else:

            representative_index = (
                len(frequencies)
                // 2
            )

            dominant_cwt_frequency = (
                frequencies[
                    representative_index
                ]
            )

            representative_scale = (
                scales[
                    representative_index
                ]
            )

        wavelet_csv_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_cwt_wavelet.csv"
        )

        wavelet_plot_path = os.path.join(
            output_dir,
            f"{fname}_{axis}_cwt_wavelet.png"
        )

        save_and_plot_wavelet(
            fs,
            representative_scale,
            dominant_cwt_frequency,
            axis,
            wavelet_csv_path,
            wavelet_plot_path
        )

        # ====================================================
        # COMBINED METRICS
        # ====================================================

        row = {

            "axis":
                axis,

            "Fs_Hz":
                fs,

            "Nyquist_Hz":
                nyquist,

            # ------------------------------------------------
            # FFT
            # ------------------------------------------------

            "FFT_DominantFrequency_Hz":
                fft_metrics[
                    "DominantFrequency_Hz"
                ],

            "FFT_DominantAmplitude_m_s2":
                fft_metrics[
                    "DominantAmplitude"
                ],

            # ------------------------------------------------
            # PSD
            # ------------------------------------------------

            "PSD_SpectralCentroid_Hz":
                fft_metrics[
                    "SpectralCentroid_Hz"
                ],

            "PSD_SpectralBandwidth_Hz":
                fft_metrics[
                    "SpectralBandwidth_Hz"
                ],

            "PSD_SpectralRMS_m_s2":
                fft_metrics[
                    "SpectralRMS"
                ],

            "PSD_TotalSpectralPower":
                fft_metrics[
                    "TotalSpectralPower"
                ],

            "PSD_F95_Hz":
                fft_metrics[
                    "F95_Hz"
                ],

            # ------------------------------------------------
            # CWT
            # ------------------------------------------------

            "CWT_DominantFrequency_Hz":
                cwt_metrics[
                    "CWT_DominantFrequency_Hz"
                ],

            "CWT_DominantAmplitude":
                cwt_metrics[
                    "CWT_DominantAmplitude"
                ],

            "CWT_GlobalWaveletPower":
                cwt_metrics[
                    "CWT_GlobalWaveletPower"
                ],

            "CWT_MeanWaveletPower":
                cwt_metrics[
                    "CWT_MeanWaveletPower"
                ]
        }

        metric_rows.append(
            row
        )

    # ========================================================
    # SAVE PER-TRIAL METRICS
    # ========================================================

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_path = os.path.join(
        output_dir,
        f"{fname}_frequency_domain_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    return metrics_df


# ============================================================
# MAIN
# ============================================================

def main(root="."):

    root = os.path.abspath(
        root
    )

    out_root = os.path.join(
        root,
        "frequency_domain_analysis"
    )

    os.makedirs(
        out_root,
        exist_ok=True
    )

    combined_rows = []

    # ========================================================
    # PROCESS ALL EXPERIMENTAL CLASSES
    # ========================================================

    for cls in CLASS_DIRS:

        cls_dir = os.path.join(
            root,
            cls
        )

        if not os.path.isdir(
            cls_dir
        ):

            print(
                f"Skipping missing folder: {cls}"
            )

            continue

        # ----------------------------------------------------
        # Class output directory
        # ----------------------------------------------------

        out_cls_dir = os.path.join(
            out_root,
            cls
        )

        os.makedirs(
            out_cls_dir,
            exist_ok=True
        )

        csv_files = sorted(
            glob.glob(
                os.path.join(
                    cls_dir,
                    "*.csv"
                )
            )
        )

        class_rows = []

        print(
            "\n=========================================="
        )

        print(
            f"Analyzing class: {cls}"
        )

        print(
            f"Trials found: {len(csv_files)}"
        )

        print(
            "=========================================="
        )

        # ====================================================
        # PROCESS EACH TRIAL
        # ====================================================

        for f in csv_files:

            # ------------------------------------------------
            # Get exact input filename without extension
            #
            # Example:
            # servo_only_trial01.csv
            #
            # becomes:
            # servo_only_trial01/
            # ------------------------------------------------

            fname = os.path.splitext(
                os.path.basename(f)
            )[0]

            print(
                f"\n  Trial: {fname}"
            )

            # ------------------------------------------------
            # Dedicated trial directory
            # ------------------------------------------------

            trial_out_dir = os.path.join(
                out_cls_dir,
                fname
            )

            os.makedirs(
                trial_out_dir,
                exist_ok=True
            )

            try:

                report_df = analyze_file(
                    f,
                    trial_out_dir
                )

            except Exception as e:

                print(
                    f"  ERROR: "
                    f"{os.path.basename(f)}"
                )

                print(
                    f"  {type(e).__name__}: {e}"
                )

                continue

            # ------------------------------------------------
            # Add trial and class identifiers
            # ------------------------------------------------

            for _, row in (
                report_df.iterrows()
            ):

                rec = {

                    "class":
                        cls,

                    "file":
                        os.path.basename(f),

                    **row.to_dict()
                }

                combined_rows.append(
                    rec
                )

                class_rows.append(
                    row.to_dict()
                )

        # ====================================================
        # CLASS SUMMARY
        # ====================================================

        if class_rows:

            cls_df = pd.DataFrame(
                class_rows
            )

            metric_columns = [
                col
                for col in cls_df.columns
                if col != "axis"
            ]

            summary = (
                cls_df
                .groupby("axis")
                [metric_columns]
                .agg(
                    [
                        "mean",
                        "std",
                        "min",
                        "max"
                    ]
                )
            )

            summary.columns = [
                "_".join(
                    map(
                        str,
                        c
                    )
                )
                for c in summary.columns
            ]

            summary = (
                summary
                .reset_index()
            )

            summary_path = os.path.join(
                out_root,
                f"summary_{cls}.csv"
            )

            summary.to_csv(
                summary_path,
                index=False
            )

            print(
                f"\nWrote "
                f"summary_{cls}.csv"
            )

    # ========================================================
    # ALL TRIALS COMBINED
    # ========================================================

    if combined_rows:

        combined_df = pd.DataFrame(
            combined_rows
        )

        combined_path = os.path.join(
            out_root,
            "all_trials_combined.csv"
        )

        combined_df.to_csv(
            combined_path,
            index=False
        )

        print(
            "\nWrote all_trials_combined.csv"
        )

    # ========================================================
    # TOP-LEVEL CSVs / STATIONARY BENCHMARK
    # ========================================================

    top_csvs = sorted(
        glob.glob(
            os.path.join(
                root,
                "*.csv"
            )
        )
    )

    stationary_files = [
        f
        for f in top_csvs
        if not any(
            keyword
            in os.path.basename(f).lower()
            for keyword in SKIP_KEYWORDS
        )
    ]

    if stationary_files:

        stat_out_dir = os.path.join(
            out_root,
            "stationary"
        )

        os.makedirs(
            stat_out_dir,
            exist_ok=True
        )

        print(
            "\n=========================================="
        )

        print(
            "Analyzing stationary benchmark(s)"
        )

        print(
            "=========================================="
        )

        for f in stationary_files:

            # ------------------------------------------------
            # Dedicated stationary file directory
            #
            # Example:
            #
            # stationary/
            #     Stationary_benchmark/
            # ------------------------------------------------

            fname = os.path.splitext(
                os.path.basename(f)
            )[0]

            stationary_trial_dir = os.path.join(
                stat_out_dir,
                fname
            )

            os.makedirs(
                stationary_trial_dir,
                exist_ok=True
            )

            print(
                f"\n  File: "
                f"{os.path.basename(f)}"
            )

            try:

                analyze_file(
                    f,
                    stationary_trial_dir
                )

            except Exception as e:

                print(
                    f"  Skipping "
                    f"{os.path.basename(f)}"
                )

                print(
                    f"  {type(e).__name__}: {e}"
                )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "Frequency-domain analysis COMPLETE."
    )

    print(
        f"Results written under:\n"
        f"{out_root}"
    )

    print(
        "=========================================="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    root_arg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "."
    )

    main(
        root_arg
    )