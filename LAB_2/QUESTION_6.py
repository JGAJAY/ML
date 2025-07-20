# Raw data
C1 = [20, 6, 2, 386]
C2 = [16, 3, 6, 289]

# Calculate magnitudes
mag1 = sum([x**2 for x in C1]) ** 0.5
mag2 = sum([x**2 for x in C2]) ** 0.5

# Normalize vectors
C1_norm = [x / mag1 for x in C1]
C2_norm = [x / mag2 for x in C2]

# Dot product of normalized vectors = cosine similarity
dot = 0
for i in range(len(C1)):
    dot += C1_norm[i] * C2_norm[i]

print("Version 2: Normalized Vectors")
print("Cosine Similarity:", round(dot, 4))
