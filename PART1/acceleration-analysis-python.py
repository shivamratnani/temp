import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('ACCELERATION.csv')


plt.figure(figsize=(12, 12))


plt.subplot(3, 1, 1)
plt.plot(df['timestamp'], df['acceleration'], label='Actual Acceleration', color='blue')
plt.plot(df['timestamp'], df['noisyacceleration'], label='Noisy Acceleration', color='red')
plt.title('Acceleration vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.grid(True)
plt.legend()


time_step = 0.1
velocity_actual = np.cumsum(df['acceleration']) * time_step
velocity_noisy = np.cumsum(df['noisyacceleration']) * time_step


plt.subplot(3, 1, 2)
plt.plot(df['timestamp'], velocity_actual, label='Velocity from Actual Acceleration', color='blue')
plt.plot(df['timestamp'], velocity_noisy, label='Velocity from Noisy Acceleration', color='red')
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()


distance_actual = np.cumsum(velocity_actual) * time_step
distance_noisy = np.cumsum(velocity_noisy) * time_step


plt.subplot(3, 1, 3)
plt.plot(df['timestamp'], distance_actual, label='Distance from Actual Acceleration', color='blue')
plt.plot(df['timestamp'], distance_noisy, label='Distance from Noisy Acceleration', color='red')
plt.title('Distance vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


print("\nResults:")
print(f"Final distance (actual acceleration): {distance_actual[-1]:.2f} meters")
print(f"Final distance (noisy acceleration): {distance_noisy[-1]:.2f} meters")
print(f"Difference between estimates: {abs(distance_actual[-1] - distance_noisy[-1]):.2f} meters")
