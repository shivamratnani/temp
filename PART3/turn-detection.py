import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import csv

def load_and_prepare_data(filename):
    
    data = []
    with open(filename, 'r') as file:

        header = next(file).strip().split(',')
        print(f"Header: {header}")
        

        for line in file:
            fields = line.strip().split(',')
            if len(fields) >= 7: 
                try:
                    row = {
                        'timestamp': float(fields[0]),
                        'gyro_z': float(fields[6])
                    }
                    data.append(row)
                except (ValueError, IndexError) as e:
                    continue  
    
    if not data:
        raise ValueError("No valid data rows found in file")
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} valid rows")
    return df

def detect_turns(df, z_threshold=0.05, min_samples_between_turns=20):
    """Detect turns using gyroscope z-axis data."""
    print("\nStarting turn detection...")
    
    gyro_z = df['gyro_z'].values
    timestamps = df['timestamp'].values
    

    dt = np.diff(timestamps) / 1e9  
    dt = np.append(dt, dt[-1])
    

    window = 5
    weights = np.ones(window) / window
    smoothed_gyro = np.convolve(gyro_z, weights, mode='same')
    
    turns = []
    turn_direction = []
    last_turn_idx = -min_samples_between_turns
    accumulated_angle = 0
    
    print(f"Processing {len(smoothed_gyro)} samples...")
    print(f"Gyro Z range: {np.min(gyro_z):.3f} to {np.max(gyro_z):.3f}")
    

    for i in range(len(df)):
        if abs(smoothed_gyro[i]) > z_threshold:
            accumulated_angle += smoothed_gyro[i] * dt[i]
        else:
            if abs(accumulated_angle) > np.pi/6:  
                if i - last_turn_idx > min_samples_between_turns:
                    turns.append(i)
                    turn_direction.append(np.sign(accumulated_angle))
                    last_turn_idx = i
            accumulated_angle = 0
    
    print(f"Found {len(turns)} turns")
    return turns, turn_direction, smoothed_gyro

def plot_results(df, smoothed_data, turns, turn_direction):
    """Plot the gyroscope data and detected turns."""
    plt.figure(figsize=(15, 6))
    

    plt.plot(df['gyro_z'].values, 'b-', alpha=0.5, label='Raw gyro-z')
    plt.plot(smoothed_data, 'g-', label='Smoothed gyro-z')
    

    used_labels = set()
    for idx, direction in zip(turns, turn_direction):
        color = 'r' if direction > 0 else 'g'
        label = 'CW Turn' if direction > 0 else 'CCW Turn'
        if label not in used_labels:
            plt.plot(idx, smoothed_data[idx], color + 'o', markersize=10, label=label)
            used_labels.add(label)
        else:
            plt.plot(idx, smoothed_data[idx], color + 'o', markersize=10)
    
    plt.title('Turn Detection Results')
    plt.xlabel('Sample')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    try:

        df = load_and_prepare_data('TURNING.csv')
        

        turns, turn_direction, smoothed_gyro = detect_turns(df)
        

        print("\nTurn Detection Results:")
        for i, (idx, direction) in enumerate(zip(turns, turn_direction)):
            turn_type = "clockwise" if direction > 0 else "counter-clockwise"
            angle = "90" if direction > 0 else "-90"
            print(f"Turn {i+1}: {turn_type} ({angle}°) at sample {idx}")
        

        plot_results(df, smoothed_gyro, turns, turn_direction)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
