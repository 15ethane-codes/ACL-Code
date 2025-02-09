import numpy as np
import matplotlib.pyplot as plt

bias = 0
slope = 0.5
num_points = 4

x_values = np.arange(num_points)
y_values = bias + slope * x_values
num_slopes = 5
test_slopes = np.linspace(0, 4*slope, num_slopes)
print(test_slopes)

def cost_function(bias, slope, x_values, y_values):
    y_predicted = bias + slope * x_values
    mse = np.mean((y_values - y_predicted) ** 2)
    return mse

mse_values = []
for slope in test_slopes:
    mse = cost_function(bias, slope, x_values, y_values)
    mse_values.append(mse)

print(mse_values)

plt.figure(figsize=(8, 6))
plt.plot(test_slopes, mse_values, 'o-', label="MSE for different slopes")
plt.xlabel('Slope (W1)')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. Slope')
plt.legend()
plt.grid(True)
plt.show()

#Predicted value: change to get accurate model
selected_slope = 1
predicted_y_values = bias + selected_slope * x_values

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, 'o', label='Actual Y values')
plt.plot(x_values, predicted_y_values, 'x-', label=f'Predicted Y values (Slope = {selected_slope})')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.title('Y-Values (Actual vs. Predicted)')
plt.legend()
plt.grid(True)
plt.show()
