import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Drag coeff, radius (m), area (m2), and mass (kg).
# c = 0.3
# r = 0.0366
# A = np.pi * r**2
# m = .15
#Chose a professionally hit soccer ball at sea level
c = 0.25
r = 0.155
A = np.pi * r**2
m = 0.43
#Air density (kg.m-3), accleration due to gravity (m.s-2).
#rho_air = 1.28
rho_air = 1.225
g = 9.81
#Define the constant
k = .3 * c * rho_air * A

#Inital speed and launch angle (from the horizontal).
v0 = 30 #Good speed for ball
phi0 = np.radians(30) #Generally the optimal angle to kick at a distance

def deriv(t, u):
    x, xdot, y, ydot = u
    speed= np.hypot(xdot, ydot)
    xdotdot = -k/m * speed * xdot
    ydotdot = -k/m * speed * ydot - g
    return xdot, xdotdot, ydot, ydotdot

#Intial conditions: x0, v0_x, y0, v0_y.
u0 = 0, v0 * np.cos(phi0), 0., v0 * np.sin(phi0)
#Integrate up to tf unless we hit the target sooner.
t0, tf = 0, 50

def hit_target(t, u):
    #We've hit the target if the y-coord is 0
    return u[2]
#Stop integration when we hit the target.
hit_target.terminal = True
#We must be moving downwards (don't stop before we begin moving upwards)
hit_target.direction = -1

def max_height(t, u):
    #Max height is obtained if y-velocity is zero.
    return u[3]

soln = solve_ivp(deriv, (t0, tf), u0, dense_output=True, events=(hit_target, max_height))
#Find time for object to land w/o air resistance
t = ((2 * v0) * np.sin(phi0)) / g
t1 = np.linspace(0, t, num=100)

    #Calcualte trajectory with no air resistance
xnd = ((v0 * t1) * np.cos(phi0))
ynd = ((v0 * t1) * np.sin(phi0)) - ((0.5 * g) * (t1**2))

print(soln)
print('Time to target = {:.2f} s'.format(soln.t_events[0][0]))
print('Time to highest point = {:.2f} s'.format(soln.t_events[1][0]))

#Grid of time points from 0 until impact time with air resistance
t = np.linspace(0, soln.t_events[0][0], 100)
#Get solution for time grid and plot trajectory with air resistance and w/o it
sol = soln.sol(t)
x, y = sol[0], sol[2]
print('Range to target, xmax = {:2f} m'.format(x[-1]))
print('Maximum height, ymax = {:2f} m'.format(max(y)))

plt.plot(x, y, 'blue', xnd, ynd, 'red')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()
