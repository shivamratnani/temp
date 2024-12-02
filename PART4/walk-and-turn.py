import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def load_data(filename):
    """Load and prepare the walking and turning data."""
    df = pd.read_csv(filename)
    df['seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9
    return df

def smooth_signals(df):
    """Smooth accelerometer and gyroscope signals using Savitzky-Golay filter."""
    window_length = 21
    poly_order = 3
    
    smoothed_data = df.copy()
    smoothed_data['gyro_z_smooth'] = savgol_filter(df['gyro_z'], window_length, poly_order)
    

    smoothed_data['accel_mag'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    smoothed_data['accel_mag_smooth'] = savgol_filter(smoothed_data['accel_mag'], window_length, poly_order)
    
    return smoothed_data

def detect_steps(smoothed_data):
    """Detect steps using magnitude of acceleration."""
    magnitude = smoothed_data['accel_mag_smooth'].values
    

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
    
    return steps

def detect_turns(smoothed_data):
    """Detect turns using gyroscope z-axis data with improved peak detection."""
    gyro_z = smoothed_data['gyro_z'].values 
    timestamps = smoothed_data['seconds'].values
    

    z_threshold = 10.0  
    min_samples_between_turns = 50 
    
    turns = []
    turn_angles = []
    last_turn_end = -min_samples_between_turns
    
    i = 0
    while i < len(gyro_z) - 1:
        if i < last_turn_end + min_samples_between_turns:
            i += 1
            continue
        

        if abs(gyro_z[i]) > z_threshold:

            turn_start = i
            accumulated_angle = 0
            peak_value = gyro_z[i]
            

            while i < len(gyro_z) - 1 and abs(gyro_z[i]) > 2.0:
                dt = timestamps[i + 1] - timestamps[i]
                accumulated_angle += gyro_z[i] * dt
                if abs(gyro_z[i]) > abs(peak_value):
                    peak_value = gyro_z[i]
                i += 1
            

            if abs(peak_value) > z_threshold:
                turns.append(turn_start)
                angle = 90 if peak_value > 0 else -90
                turn_angles.append(angle)
                last_turn_end = i
        
        i += 1
    
    return turns, turn_angles

def plot_trajectory(steps, turns, turn_angles, step_length=1.0):
    """Plot the walking trajectory."""
    x, y = 0, 0
    direction = 90 
    positions_x = [x]
    positions_y = [y]
    

    events = [(step_idx, 'step', 0) for step_idx in steps] + \
            [(turn_idx, 'turn', angle) for turn_idx, angle in zip(turns, turn_angles)]
    events.sort(key=lambda x: x[0])
    

    current_direction = direction
    for _, event_type, angle in events:
        if event_type == 'step':
            angle_rad = np.radians(current_direction)
            x += step_length * np.cos(angle_rad)
            y += step_length * np.sin(angle_rad)
        else: 
            current_direction += angle

            current_direction = current_direction % 360
        
        positions_x.append(x)
        positions_y.append(y)
    

    plt.figure(figsize=(10, 10))
    plt.plot(positions_x, positions_y, 'b-', label='Walking path')
    plt.plot(positions_x[0], positions_y[0], 'go', label='Start')
    plt.plot(positions_x[-1], positions_y[-1], 'ro', label='End')
    

    for i in range(0, len(positions_x)-1, max(1, len(positions_x)//20)):
        dx = positions_x[i+1] - positions_x[i]
        dy = positions_y[i+1] - positions_y[i]
        plt.arrow(positions_x[i], positions_y[i], dx*0.5, dy*0.5,
                 head_width=0.2, head_length=0.3, fc='gray', ec='gray', alpha=0.5)
    
    plt.title('Walking Trajectory')
    plt.xlabel('X Distance (meters)')
    plt.ylabel('Y Distance (meters)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

def plot_sensor_data(smoothed_data, turns):
    """Plot gyroscope data with detected turns."""
    plt.figure(figsize=(15, 5))
    plt.plot(smoothed_data['seconds'], smoothed_data['gyro_z'], 
             label='Gyro-z', linewidth=2)
    

    if turns:
        plt.plot(smoothed_data['seconds'].iloc[turns], 
                 smoothed_data['gyro_z'].iloc[turns], 
                 'ro', label='Detected turns', markersize=10)
    
    plt.title('Gyroscope Z-axis Data with Detected Turns')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angular velocity (rad/s)')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():

    df = load_data('WALKING_AND_TURNING.csv')
    
    smoothed_data = smooth_signals(df)
    

    steps = detect_steps(smoothed_data)
    
    turns, turn_angles = detect_turns(smoothed_data)
    

    print(f"\nResults:")
    print(f"Detected {len(steps)} steps")
    print(f"Detected {len(turns)} turns")
    print("\nTurn angles:")
    for i, angle in enumerate(turn_angles):
        print(f"Turn {i+1}: {angle:.1f} degrees")
    

    plot_sensor_data(smoothed_data, turns)
    

    plot_trajectory(steps, turns, turn_angles)

if __name__ == "__main__":
    main()
