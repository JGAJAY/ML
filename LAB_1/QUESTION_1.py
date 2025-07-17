def count_pairs(lst, target=10):
    count = 0
    seen = set()
    for num in lst:
        if (target - num) in seen:
            count += 1
        seen.add(num)
    return count

lst = [2, 7, 4, 1, 3, 6]
print("Number of pairs with sum 10:", count_pairs(lst))
