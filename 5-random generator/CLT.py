import random
import matplotlib.pyplot as plt

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

