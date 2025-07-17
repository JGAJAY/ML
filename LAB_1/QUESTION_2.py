def calculate(lst):
    if len(lst) < 3:
        return "Range determination not possible"
    return max(lst) - min(lst)

lst = [5, 3, 8, 1, 0, 4]
print("Range of list:", calculate(lst))
