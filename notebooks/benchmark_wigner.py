import numpy as np
from matplotlib_params_file import *
# Wigner3j code
from timeit import default_timer as timer
import fast_Wigner3j as Wigner3j
import time

def time_execution(lmax_start, lmax_end, step, iterations):
    lmax_values = []
    mean_times = []
    std_dev_times = []

    for lmax in range(lmax_start, lmax_end + step, step):
        times = []
        wl = np.random.rand(lmax)
        # Initialize the CoupleMat class
        couple_mat = Wigner3j.CoupleMat(lmax, wl)

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
        std_dev_time = np.std(times)
        
        # Store results
        lmax_values.append(lmax)
        mean_times.append(mean_time)
        std_dev_times.append(std_dev_time)
        
        # Print the mean and standard deviation for each lmax
        print(f'lmax: {lmax}, mean time: {mean_time:.5f}s, std dev: {std_dev_time:.5f}s', flush=True)
    
    return lmax_values, mean_times, std_dev_times

def main():
    # Parameters for the timing function
    lmax_start = 500
    lmax_end = 10000
    step = 500
    iterations = 100
    
    # Execute the timing function
    lmax_values, mean_times, std_dev_times = time_execution(
        lmax_start, lmax_end, step, iterations
    )
    
    # Plotting the results
    plt.errorbar(lmax_values, mean_times, yerr=std_dev_times, fmt='-o',color='k', capsize=5)
    plt.xlabel(r'$l_{\text{max}}$')
    plt.ylabel('Time taken (seconds)')
    # plt.title('Execution Time vs. Number of l values')
    plt.legend()
    # plt.grid(True)
    plt.savefig('benchmark_wigner.pdf', bbox_inches='tight')

# Ensure the main function is called when this script is run
if __name__ == "__main__":
    main()
