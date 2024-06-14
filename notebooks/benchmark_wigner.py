import numpy as np
from matplotlib_params_file import *
# Wigner3j code
from timeit import default_timer as timer
import fast_Wigner3j as Wigner3j
import time

import sys
sys.path.insert(0, '/Users/rdb/Desktop/research/lya/P3D_Cell/directsht-lya/')
from sht.mask_deconvolution import MaskDeconvolution

def time_execution(lmax_start, lmax_end, step, iterations):
    lmax_values = []
    mean_times = []
    # std_dev_times = []

    for lmax in range(lmax_start, lmax_end + step, step):
        times = []
        wl = np.random.rand(lmax)
        # Initialize the CoupleMat class
        couple_mat = Wigner3j.CoupleMat(lmax, wl)
        coupling_matrix = couple_mat.compute_matrix()

        for _ in range(iterations):
            # Time the execution of compute_matrix method
            start = time.time()
            coupling_matrix = couple_mat.compute_matrix()
            end = time.time()
            
            # Calculate elapsed time
            elapsed_time = end - start
            times.append(elapsed_time)
        
        # Calculate mean and standard deviation
        mean_time = np.mean(times)
        # std_dev_time = np.std(times)
        
        # Store results
        lmax_values.append(lmax)
        mean_times.append(mean_time)
        # std_dev_times.append(std_dev_time)
        
        # Print the mean and standard deviation for each lmax
        print(f'lmax: {lmax}, mean time: {mean_time:.5f}s', flush=True)
    
    return lmax_values, mean_times


def time_execution_SHT(lmax_start, lmax_end, step, iterations):
    lmax_values = []
    mean_times = []
    # std_dev_times = []

    for lmax in range(lmax_start, lmax_end + step, step):
        times = []
        wl = np.random.rand(lmax)

        for _ in range(iterations):
            # Time the execution of compute_matrix method
            start = time.time()
            MD = MaskDeconvolution(lmax,wl)
            end = time.time()
            
            # Calculate elapsed time
            elapsed_time = end - start
            times.append(elapsed_time)
        
        # Calculate mean and standard deviation
        mean_time = np.mean(times)
        # std_dev_time = np.std(times)
        
        # Store results
        lmax_values.append(lmax)
        mean_times.append(mean_time)
        # std_dev_times.append(std_dev_time)
        
        # Print the mean and standard deviation for each lmax
        print(f'lmax: {lmax}, mean time: {mean_time:.5f}s', flush=True)
    
    return lmax_values, mean_times#, std_dev_times

def main():
    # Parameters for the timing function
    lmax_start = 100
    lmax_end = 1000
    step = 100
    iterations = 1
    
    # Execute the timing function
    lmax_values_SHT, mean_times_SHT = time_execution_SHT(
        lmax_start, lmax_end, step, iterations
    )

    lmax_values, mean_times = time_execution(
        lmax_start, lmax_end, step, iterations
    )

    
    np.savetxt('benchmark_wigner.csv', np.array([lmax_values, mean_times, mean_times_SHT]).T, delimiter=',', header='lmax,mean,std_dev,mean_SHT', comments='')

    # # Plotting the results
    # plt.errorbar(lmax_values, mean_times, yerr=std_dev_times, fmt='-o',color='k', capsize=5)
    # plt.xlabel(r'$\ell_{\mathrm{max}}$')
    # plt.ylabel('Time taken (seconds)')
    # # plt.title('Execution Time vs. Number of l values')
    # plt.legend()
    # # plt.grid(True)
    # plt.savefig('benchmark_wigner.pdf', bbox_inches='tight')

    # Plotting the results
    fig, ax1 = plt.subplots()

    # Plot the first set of data
    ax1.plot(lmax_values, mean_times, 'ko-', label='This work')
    ax1.set_xlabel(r'$\ell_{\mathrm{max}}$')
    ax1.set_ylabel(r'Time $t$ [sec]', color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    # Create a second y-axis to plot the second set of data
    ax2 = ax1.twinx()
    ax2.plot(lmax_values_SHT, mean_times_SHT, 'C0o--', label='Recurrence relation')
    ax2.set_ylabel(r'Time $t$ [sec]', color='C0')
    ax2.tick_params(axis='y', labelcolor='C0')

    fig.tight_layout()  # Adjust the layout to make room for the second y-axis

    # Add legends for both plots
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 0.9))

    # plt.title('Execution Time vs. Number of l values')
    # plt.grid(True)
    plt.savefig('benchmark_wigner_compare.pdf', bbox_inches='tight')
    plt.show()

# Ensure the main function is called when this script is run
if __name__ == "__main__":
    main()
