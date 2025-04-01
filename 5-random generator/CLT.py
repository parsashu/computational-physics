import random
import matplotlib.pyplot as plt
import numpy as np

N_nums = 1000
N_trials = 10000

def CLT(N_nums, N_trials):
    sum_list = []
    
    for _ in range(N_trials):
        numbers = []
        for _ in range(N_nums):
            numbers.append(random.randint(0, 9))       
        sum_list.append(sum(numbers))
        
    return sum_list


    
sum_list = CLT(N_nums, N_trials)

plt.figure(figsize=(10, 6))
plt.hist(sum_list, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel("Sum of Random Numbers")
plt.ylabel("Frequency")
plt.title(f"Distribution of Sums of {N_nums} Random Numbers (0-9) over {N_trials} Trials")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


def calc_w(sum_list):
    sum_list = np.array(sum_list)
    mean = np.mean(sum_list)
    mean_square = np.mean(sum_list**2)
    w = (mean_square - mean**2)**0.5
    return w


N_nums_range = range(10, 1000, 100)
w_list = []

for N_nums in N_nums_range:
    sum_list = CLT(N_nums, N_trials)
    w_list.append(calc_w(sum_list))


plt.figure(figsize=(10, 6))
plt.plot(N_nums_range, w_list, 'o', markersize=3, label='Measured width')

x_sqrt = np.sqrt(N_nums_range)
coeff = np.polyfit(x_sqrt, w_list, 1)[0]
fitted_curve = coeff * np.sqrt(N_nums_range)
plt.plot(N_nums_range, fitted_curve, 'r--', label=f'{coeff:.4f}·√N')

plt.xlabel('N (Number of Random Numbers)')
plt.ylabel('Width (w)')
plt.title('Width of Distribution vs Number of Random Numbers')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()





