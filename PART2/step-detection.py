import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def load_and_prepare_data(filename):

    df = pd.read_csv(filename)
    df['seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9
    return df

def smooth_signals(df):

    window_length = 21  
    poly_order = 3
    
    smoothed_data = pd.DataFrame()
    for axis in ['accel_x', 'accel_y', 'accel_z']:
        smoothed_data[f'{axis}_smooth'] = savgol_filter(df[axis], window_length, poly_order)
    
    return smoothed_data

def detect_steps(df, smoothed_data):


    magnitude = np.sqrt(
        smoothed_data['accel_x_smooth']**2 + 
        smoothed_data['accel_y_smooth']**2 + 
        smoothed_data['accel_z_smooth']**2
    )
    

    min_peak_height = 10.5
    min_samples_between_peaks = 20
    

    steps = []
    last_peak = 0
    
    for i in range(1, len(magnitude)-1):
        if i < last_peak + min_samples_between_peaks:
            continue
            
        if (magnitude[i] > min_peak_height and 
            magnitude[i] > magnitude[i-1] and 
            magnitude[i] > magnitude[i+1]):
            steps.append(i)
            last_peak = i
    

    actual_steps = steps[:len(steps)//2] 
    
    return actual_steps, magnitude

def plot_results(df, smoothed_data, steps, magnitude):

    plt.figure(figsize=(15, 10))
    

    plt.subplot(2, 1, 1)
    plt.plot(df['seconds'], df['accel_x'], label='X', alpha=0.5)
    plt.plot(df['seconds'], df['accel_y'], label='Y', alpha=0.5)
    plt.plot(df['seconds'], df['accel_z'], label='Z', alpha=0.5)
    plt.title('Raw Acceleration Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True)
    

    plt.subplot(2, 1, 2)
    plt.plot(df['seconds'], magnitude, label='Magnitude', color='blue')
    plt.plot(df['seconds'].iloc[steps], magnitude.iloc[steps], 'ro', label='Detected Steps')
    plt.title(f'Acceleration Magnitude with Detected Steps (Total: {len(steps)} steps)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration Magnitude (m/s²)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():

    df = load_and_prepare_data('WALKING.csv')
    smoothed_data = smooth_signals(df)
    steps, magnitude = detect_steps(df, smoothed_data)
    

    print(f"Number of steps detected: {len(steps)}")
    

    plot_results(df, smoothed_data, steps, magnitude)

if __name__ == "__main__":
    main()
