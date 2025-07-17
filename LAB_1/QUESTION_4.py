from collections import Counter
import re

def highest_occurrence_char(s):
    s = re.sub(r'[^a-zA-Z]', '', s)  
    counter = Counter(s.lower())
    char, freq = counter.most_common(1)[0]
    return char, freq

s = "hippopotamus"
char, freq = highest_occurrence_char(s)
print(f"Highest occurring character: '{char}' with count: {freq}")
