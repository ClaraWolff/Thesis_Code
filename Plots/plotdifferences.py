import pickle
import matplotlib.pyplot as plt
import itertools

def load_data(file_path):
    with open(file_path, "rb") as f:
        differences = pickle.load(f)

    data = [
        (
            entry["number node"],
            entry["revisits after"],
            entry["penalty after"],
            entry["day"]
        )
        for entry in differences.values()
    ]
    data.sort(key=lambda x: x[0])
    return zip(*data)  # sorted_node_nums, revisits, penalties, days


# --- File paths ---
file1 = "Saves/lambda0_0_differences_revisits.pkl"
file2 = "Saves/lambda80_10_differences_revisits.pkl"
file3 = "Saves/lambda80_25_differences_revisits.pkl"
file4 = "Saves/lambda80_50_differences_revisits.pkl"
file5 = "Saves/lambda80_100_differences_revisits.pkl"
file6 = "Saves/lambda80_1_differences_revisits.pkl"
file7 = "Saves/lambda80_0_differences_revisits.pkl"
file8 = "Saves/lambda80_1000_differences_revisits.pkl"
file9 = "Saves/lambda80_10000_differences_revisits.pkl"
file10 = "Saves/lambda80_50000_differences_revisits.pkl"

# --- Load all datasets ---
nodes1, revisits1, penalties1, days1 = load_data(file1)
nodes2, revisits2, penalties2, days2 = load_data(file2)
nodes3, revisits3, penalties3, days3 = load_data(file3)
nodes4, revisits4, penalties4, days4 = load_data(file4)
nodes5, revisits5, penalties5, days5 = load_data(file5)
nodes6, revisits6, penalties6, days6 = load_data(file6)
nodes7, revisits7, penalties7, days7 = load_data(file7)
nodes8, revisits8, penalties8, days8 = load_data(file8)
nodes9, revisits9, penalties9, days9 = load_data(file9)
nodes10, revisits10, penalties10, days10 = load_data(file10)

# --- Identify day change points by index ---
day_starts = []
day_labels = []
last_day = None
for i, day in enumerate(days1):  # Use index, since plot is index-based
    if day != last_day:
        day_starts.append(i)
        day_labels.append(day)
        last_day = day

# --- Plot Revisit Comparison ---
plt.figure(figsize=(12, 8))
plt.plot(range(len(revisits1)), revisits1, label="lambda = (0,0)", color='blue')
plt.plot(range(len(revisits6)), revisits6, label="lambda = (12,1)", color='pink')
plt.plot(range(len(revisits2)), revisits2, label="lambda = (12,10)", color='green')
plt.plot(range(len(revisits3)), revisits3, label="lambda = (12,25)", color='red')
plt.plot(range(len(revisits4)), revisits4, label="lambda = (12,50)", color='orange')
plt.plot(range(len(revisits5)), revisits5, label="lambda = (12,100)", color='purple')


for idx, label, color in zip(day_starts, day_labels, itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])):
    plt.axvline(x=idx, linestyle='--', color=color, linewidth=1.2, label=label)

plt.axhline(y=0, color='black', linewidth=0.8)
plt.xlabel("Insertion Number")
plt.ylabel("Estimated Revisits")
#plt.title("Estimated Revisits per Node (Greedy Insertion Order)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot Penalty Comparison ---
plt.figure(figsize=(12, 8))
plt.plot(range(len(penalties1)), penalties1, label="lambda = (0,0)", color='blue')
plt.plot(range(len(penalties7)), penalties7, label="lambda = (80,0)", color='blue')
plt.plot(range(len(penalties6)), penalties6, label="lambda = (80,1)", color='pink')
plt.plot(range(len(penalties2)), penalties2, label="lambda = (80,10)", color='green')
plt.plot(range(len(penalties3)), penalties3, label="lambda = (80,25)", color='red')
plt.plot(range(len(penalties4)), penalties4, label="lambda = (80,50)", color='orange')
plt.plot(range(len(penalties5)), penalties5, label="lambda = (80,100)", color='purple')
plt.plot(range(len(penalties8)), penalties8, label="lambda = (80,1000)", color='aqua')
plt.plot(range(len(penalties9)), penalties9, label="lambda = (80,10000)", color='salmon')
plt.plot(range(len(penalties10)), penalties10, label="lambda = (80,50000)", color='lawngreen')


for idx, label, color in zip(day_starts, day_labels, itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])):
    plt.axvline(x=idx, linestyle='--', color=color, linewidth=1.2, label=label)

plt.axhline(y=0, color='black', linewidth=0.8)
plt.xlabel("Insertion Number")
plt.ylabel("Penalty (FTR Weighted)")
plt.legend()
plt.rcParams.update({'font.size': 13})
plt.savefig("Plots/lines_penalty_80_extreme.png", dpi=300,bbox_inches="tight")
plt.show()


import pickle

# Load differences
with open("Saves/lambda12_10_differences_revisits.pkl", "rb") as f:
    differences = pickle.load(f)

# Reconstruct expected num values
max_num = max(differences.keys())  # assume nums are 1-based and continuous up to max
expected_nums = set(range(1, max_num + 1))
saved_nums = set(differences.keys())
missing_nums = expected_nums - saved_nums

print(f"Expected num values: 1 to {max_num}")
print(f"Saved: {len(saved_nums)}")
print(f"Missing: {len(missing_nums)}")
print(f"Missing num values: {sorted(missing_nums)}")
