C1 = [20, 6, 2, 386]
C2 = [16, 3, 6, 289]

binary_C1 = [1 if val != 0 else 0 for val in C1]
binary_C2 = [1 if val != 0 else 0 for val in C2]

f11 = f10 = f01 = f00 = 0
for i in range(len(C1)):
    if binary_C1[i] == 1 and binary_C2[i] == 1:
        f11 += 1
    elif binary_C1[i] == 1 and binary_C2[i] == 0:
        f10 += 1
    elif binary_C1[i] == 0 and binary_C2[i] == 1:
        f01 += 1
    else:
        f00 += 1

jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) else 0
smc = (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) else 0

print("Binary C1:", binary_C1)
print("Binary C2:", binary_C2)
print("JC:", round(jc, 4), "SMC:", round(smc, 4))
