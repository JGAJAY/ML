import random

def mean(lst):
    total = 0
    for num in lst:
        total += num
    return total / len(lst)

def median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
    else:
        return sorted_lst[mid]

def mode(lst):
    freq = {}
    max_count = 0
    mode_val = lst[0]
    for num in lst:
        freq[num] = freq.get(num, 0) + 1
        if freq[num] > max_count:
            max_count = freq[num]
            mode_val = num
    return mode_val

random_numbers = [random.randint(1, 10) for _ in range(25)]

print("Random Numbers:", random_numbers)
print("Mean:", mean(random_numbers))
print("Median:", median(random_numbers))
print("Mode:", mode(random_numbers))
