import os
import re
import json
import subprocess
import time
import pandas as pd

ini_file = "VSE/racesim/input/parameters/pars_Spielberg_2019.ini"
main_script = "main_racesim.py"

drivers = ["HAM", "RIC", "MAG", "HUL", "KVY", "RAI", "VET", "BOT", "PER", "GRO", "SAI", "VER", "GIO", "STR", "GAS", "LEC", "NOR", "ALB", "RUS", "KUB"]

def set_vse_type(strategies):
    with open(ini_file, 'r') as f:
        content = f.read()
    
    vse_dict_str = json.dumps(strategies)
    new_content = re.sub(r'"vse_type":\s*\{[^}]+\}', '"vse_type": ' + vse_dict_str, content)
    
    with open(ini_file, 'w') as f:
        f.write(new_content)

def get_race_times():
    racetime_file = "VSE/racesim/output/results/Spielberg_2019_racetimes.csv"
    df = pd.read_csv(racetime_file)
    times = {}
    for d in drivers:
        if d in df.columns:
            # Get the last non-null value for this driver
            valid_times = df[d].dropna()
            if not valid_times.empty:
                times[d] = float(valid_times.iloc[-1])
            else:
                times[d] = 0.0
    return times

def run_case(case_name, strategies):
    print(f"Running Case: {case_name}")
    set_vse_type(strategies)
    # Run the simulator and capture stdout
    result = subprocess.run(["python", main_script], cwd="VSE", capture_output=True, text=True)
    
    times = {}
    # Parse mean race times from stdout
    # Line format: "RESULT: HAM: Mean Pos: 4.2, Mean Race Time: 4945.834s"
    for line in result.stdout.split("\n"):
        if "RESULT:" in line and "Mean Race Time:" in line:
            parts = line.split(":")
            initials = parts[1].strip()
            # Extract time from the last part " 4945.834s"
            rt_str = parts[4].strip().replace("s", "")
            times[initials] = float(rt_str)
    
    return times

cases = [
    ("All Historical", {d: "realstrategy" for d in drivers}),
    ("Historical except HAM (Supervised)", {d: "supervised" if d == "HAM" else "realstrategy" for d in drivers}),
    ("Historical except BOT (Supervised)", {d: "supervised" if d == "BOT" else "realstrategy" for d in drivers}),
    ("All Supervised", {d: "supervised" for d in drivers})
]

results = {}
for name, strats in cases:
    results[name] = run_case(name, strats)

print("="*50)
print("MONTE CARLO RESULTS (Mean Race Times in seconds, 5 runs)")
print("="*50)

header_row = f"{'Driver':<5}" + "".join([f" | {name[:20]:<20}" for name, _ in cases])
print(header_row)

rows = []
for d in drivers:
    row_str = f"{d:<5}"
    for name, _ in cases:
        t = results[name].get(d, 0.0)
        row_str += f" | {t:<20.3f}"
    print(row_str)
    rows.append(row_str)

with open("final_results_montecarlo_5runs.txt", "w") as f:
    f.write("="*50 + "\n")
    f.write("MONTE CARLO RESULTS (Mean Race Times in seconds, 5 runs)\n")
    f.write("="*50 + "\n")
    f.write(header_row + "\n")
    for r in rows:
        f.write(r + "\n")

# Restore original ini file content just in case
set_vse_type({d: "supervised" for d in drivers})
