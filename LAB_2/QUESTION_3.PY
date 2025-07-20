import statistics

# Data set
stock_data = [
    {"date": "Jul 14, 2020", "month": "Jul", "day": "Tue", "value": 1362.15, "change": -2.52},
    {"date": "Jul 13, 2020", "month": "Jul", "day": "Mon", "value": 1397.35, "change": -0.26},
    {"date": "Jul 10, 2020", "month": "Jul", "day": "Fri", "value": 1400.95, "change": 1.15},
    {"date": "Jul 09, 2020", "month": "Jul", "day": "Thu", "value": 1385.05, "change": -0.36},
    {"date": "Jul 08, 2020", "month": "Jul", "day": "Wed", "value": 1390.10, "change": -0.52},
    {"date": "Jul 07, 2020", "month": "Jul", "day": "Tue", "value": 1397.40, "change": -0.24},
    {"date": "Jul 06, 2020", "month": "Jul", "day": "Mon", "value": 1400.75, "change": -0.31},
    {"date": "Jul 03, 2020", "month": "Jul", "day": "Fri", "value": 1405.10, "change": -0.51},
    {"date": "Jul 02, 2020", "month": "Jul", "day": "Thu", "value": 1412.35, "change": 3.62},
    {"date": "Jul 01, 2020", "month": "Jul", "day": "Wed", "value": 1363.05, "change": 0.32},
]

all_prices = [entry["value"] for entry in stock_data]
all_changes = [entry["change"] for entry in stock_data]

avg_price = statistics.mean(all_prices)
var_price = statistics.variance(all_prices)
print(f"Average Price: {avg_price:.2f}")
print(f"Price Variance: {var_price:.2f}")

wed_prices = [entry["value"] for entry in stock_data if entry["day"] == "Wed"]
avg_wed_price = statistics.mean(wed_prices)
print(f"\nWednesday Average: {avg_wed_price:.2f}")
print(f"Difference from Total Average: {avg_wed_price - avg_price:.2f}")

apr_values = [entry["value"] for entry in stock_data if entry["month"] == "Apr"]
if apr_values:
    avg_apr_price = statistics.mean(apr_values)
    print(f"\nApril Average: {avg_apr_price:.2f}")
    print(f"Difference from Total Average: {avg_apr_price - avg_price:.2f}")
else:
    print("\nNo April data found.")

loss_entries = [ch for ch in all_changes if ch < 0]
probability_loss = len(loss_entries) / len(stock_data)
print(f"\nLoss Probability: {probability_loss:.2%}")

wed_changes = [entry["change"] for entry in stock_data if entry["day"] == "Wed"]
wed_gains = [x for x in wed_changes if x > 0]
prob_wed_gain = len(wed_gains) / len(wed_changes)
print(f"\nProfit on Wednesday: {prob_wed_gain:.2%}")
print(f"Conditional P(Profit | Wednesday): {prob_wed_gain:.2%}")

print("\nText Scatter Plot (Day vs Change%):")
for entry in stock_data:
    print(f"{entry['day']:>3} : {entry['change']:>6.2f}%")
