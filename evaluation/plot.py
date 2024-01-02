import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def concatenate_dataframes(directory):
    gpt2_frames = []
    ttc_frames = []

    for filename in os.listdir(directory):
        # Load the dataframe
        full_path = os.path.join(directory, filename)
        df = pd.read_pickle(full_path)

        # Check if it's a gpt2 or ttc file and append it to the appropriate list
        if filename.startswith("gpt2"):
            gpt2_frames.append(df)
        elif filename.startswith("ttc"):
            ttc_frames.append(df)

    # Concatenate all dataframes in each list
    concatenated_gpt2 = pd.concat(gpt2_frames, ignore_index=True)
    concatenated_ttc = pd.concat(ttc_frames, ignore_index=True)

    return concatenated_gpt2, concatenated_ttc

def plot_histograms(gpt2_df, ttc_df, n_bins=50, trunc=False, show_mean=False, align=False, **kwargs):
    BINS = [8,16,32,64,128,256,512]

    gpt2_arr = gpt2_df['gpt2'].copy()
    if trunc:
        gpt2_arr = gpt2_arr[gpt2_arr != 1024]

    # Plot histogram for gpt2 values
    n, bins, patches = plt.hist(gpt2_arr, color='blue', alpha=0.7, label='GPT2', bins=n_bins, density=True, **kwargs)

    if not align:
        bins = n_bins

    if show_mean:
        gpt2_mean = np.mean(gpt2_arr)
        plt.axvline(gpt2_mean, color=patches[0].get_facecolor(), linestyle='dotted', label=f'GPT2 Mean: {gpt2_mean:.2f}')

    # Plot histograms for each key in ttc dataframes
    for key in ttc_df.columns:
        ttc_arr = ttc_df[key].copy()
        if trunc:
            ttc_arr = ttc_arr[ttc_arr != 1024]
        n, _, patches = plt.hist(ttc_arr, alpha=0.7, bins=bins, label=f'target_len={BINS[key]}', density=True, **kwargs)
        if show_mean:
            ttc_mean = np.mean(ttc_arr)
            plt.axvline(ttc_mean, color=patches[0].get_facecolor(), linestyle='dotted', label=f'TTC-{BINS[key]} Mean: {ttc_mean:.2f}')

    plt.xlabel('Generated Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Histograms of GPT2 and TTC sequence lengths')
    plt.legend()
    plt.show()


def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <base_directory>")
        sys.exit(1)

    base_directory = sys.argv[1]

    BINS = [8,16,32,64,128,256,512]

    # Concatenate dataframes
    concatenated_gpt2, concatenated_ttc = concatenate_dataframes(base_directory)

    print('== GPT2 (unfiltered) ==')
    print(concatenated_gpt2.describe())
    print('== TTC (unfiltered) ==')
    print(concatenated_ttc.describe())
    for key in concatenated_ttc.columns:
        print(f'== {BINS[key]} (filtered) ==')
        print(concatenated_ttc[key][concatenated_ttc[key]!=1024].describe())
    print(f'== GPT2 (filtered) ==')
    print(concatenated_gpt2['gpt2'][concatenated_gpt2['gpt2']!=1024].describe())

    # Plot and save histograms
    plt.figure(figsize=(12,8), dpi=300)
    plot_histograms(concatenated_gpt2, concatenated_ttc, n_bins=20)
    #plt.xscale('log')
    plt.savefig(f"./histograms.png")

    # Plot and save histograms
    plt.figure(figsize=(12,8), dpi=300)
    plot_histograms(concatenated_gpt2, concatenated_ttc, n_bins=10, trunc=True, show_mean=True)
    #plt.xscale('log')
    plt.savefig(f"./trunc_histograms.png")

    plt.figure(figsize=(12,8), dpi=300)
    plt.xlabel('Generated Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of unfiltered GPT2 sequence lengths')
    plt.hist(concatenated_gpt2['gpt2'], bins=20)
    #plt.xscale('log')
    plt.savefig(f"./gpt2_histogram_unfiltered.png")

    plt.figure(figsize=(12,8), dpi=300)
    plt.xlabel('Generated Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of filtered GPT2 sequence lengths')
    plt.hist(concatenated_gpt2['gpt2'][concatenated_gpt2['gpt2']!=1024], bins=20)
    #plt.xscale('log')
    plt.savefig(f"./gpt2_histogram_filtered.png")

if __name__ == "__main__":
    main()

