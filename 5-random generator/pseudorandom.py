def rand_LCG(N, seed=42):
    m = 2**31
    a = 1664525
    c = 1013904223
    numbers = []
    x = seed

    for _ in range(N):
        x = (a * x + c) % m
        numbers.append(x)

    numbers = [num % 10 for num in numbers]
    return numbers
