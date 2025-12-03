import numpy as np
import random
import math
import time
import os
from numba import jit #  we REALLY need this here!
import matplotlib.pyplot as plt

# Parameters
NX = 64
NY = 64
ntherm = 1000
VisualDisplay = 1
SleepTime = 300000  # in microseconds

@jit
def update_spin(nx, ny, env, spin):
    """Do a metropolis update on a spin at position (nx, ny) whose environment is env"""
    current_spin = spin[nx, ny]
    newspin = 1 if np.random.random() < 0.5 else -1
    DeltaBetaE = -(newspin - current_spin) * env
    if DeltaBetaE <= 0 or np.random.random() < math.exp(-DeltaBetaE):
        spin[nx, ny] = newspin

@jit
def sweep(beta, h, spin):
    """Sweep through all lattice sites"""
    for nx in range(1, NX + 1):
        for ny in range(1, NY + 1):
            environment = (beta * (spin[nx, ny-1] + spin[nx, ny+1] + 
                                 spin[nx-1, ny] + spin[nx+1, ny]) + h)
            update_spin(nx, ny, environment, spin)

@jit
def initialize_hot(spin):
    """Initialize lattice with random spins"""
    spin = np.zeros((NX + 2, NY + 2), dtype=np.int8)
    spin[1:-1, 1:-1] = np.where(np.random.random((NX, NY)) < 0.5, 1, -1)

@jit
def magnetization(spin):
    """Calculate average magnetization"""
    return np.mean(spin[1:-1, 1:-1])

@jit
def energy(beta,spin,h):
    inter = 0
    for nx in range(1, NX + 1):
        for ny in range(1, NY + 1):
            inter -= (spin[nx, ny-1] + spin[nx, ny+1] + 
                                 spin[nx-1, ny] + spin[nx+1, ny] + h)*spin[nx,ny]
    return inter
    

def display_lattice(T,spin):
    """Display the lattice configuration"""
    if SleepTime > 0:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # Convert spins to characters
    chars = np.where(spin[1:-1, 1:-1] == 1, 'X', '-')
    for row in chars:
        print(''.join(row))
    
    print(f"T = {T:.6f}:   magnetization <sigma> = {magnetization(spin):.6f}")
    
    if SleepTime > 0:
        time.sleep(SleepTime / 1_000_000)  # Convert microseconds to seconds
    else:
        print()

def plot_fun():
    spin = np.zeros((NX + 2, NY + 2), dtype=np.int8)
    
    nsweep = 1000
    h = 0
    Tmax = 5
    ntemp = 50
    
    initialize_hot(spin)

    temps = []
    mag = []
    e = []
    cv = []

    for itemp in range(ntemp, 0, -1):
        T = (Tmax * itemp) / ntemp
        beta = 1/T
        
        # Thermalization sweeps
        for _ in range(ntherm):
            sweep(beta, h, spin)
            
        # Main sweeps
        total_mag = 0
        E = 0
        E_square = 0
        for _ in range(nsweep):
            sweep(beta, h, spin)
            total_mag += np.sum(spin[1:-1, 1:-1])
            en = energy(beta, spin, h)
            E += en
            E_square += en**2
            
        avg_mag = total_mag / (nsweep * NX * NY)
        avg_E = E/nsweep
        avg_E_sq = E_square/nsweep
        CV = 1/(NX*NY)**2*(avg_E_sq-avg_E**2)/T/T

        temps.append(T)
        mag.append(avg_mag)
        e.append(avg_E/(NX*NY))
        cv.append(CV)

    fig, ax = plt.subplots(3, figsize=[12,9])
    ax[0].plot(temps,mag)
    ax[0].set_ylabel("magnetization")

    ax[1].plot(temps, e)
    ax[1].set_ylabel("mean energy")

    ax[2].plot(temps, cv)
    ax[2].set_ylabel("specific heat")
    ax[2].set_xlabel("Temperature")

    #plt.show()
    plt.savefig("ising.pdf")
    


    

def main():
    # Initialize the lattice with boundary spins as a global numpy array
    spin = np.zeros((NX + 2, NY + 2), dtype=np.int8)
    
    output_filename = "ising2d_vs_T.dat"
    
    print(f"Program calculate <sigma> vs. T for a 2D Ising model of "
          f"{NX}x{NY} spins with free boundary conditions.\n")
    
    np.random.seed(int(time.time()))
    
    nsweep = int(input("Enter # sweeps per temperature sample:\n"))
    h = float(input("Enter value of magnetic field parameter h:\n"))
    Tmax = float(input("Enter starting value (maximum) of temperature T (=1/beta):\n"))
    ntemp = int(input("Enter # temperatures to simulate:\n"))
    
    initialize_hot(spin)
    
    with open(output_filename, 'w') as output:
        # Do ntemp temperatures between Tmax and 0
        for itemp in range(ntemp, 0, -1):
            T = (Tmax * itemp) / ntemp
            beta = 1/T
            
            # Thermalization sweeps
            for _ in range(ntherm):
                sweep(beta, h, spin)
            
            # Main sweeps
            total_mag = 0
            for _ in range(nsweep):
                sweep(beta, h, spin)
                total_mag += np.sum(spin[1:-1, 1:-1])
            
            avg_mag = total_mag / (nsweep * NX * NY)
            output.write(f"{T:.6f} {avg_mag:.6f}\n")
            
            if VisualDisplay:
                display_lattice(T,spin)
    
    print(f"Output file is {output_filename}")

if __name__ == "__main__":
    #main()
    plot_fun()
