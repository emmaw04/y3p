import re
import json
import subprocess
import time
import pandas as pd

#the path to the input parameters for the 2019 austrian grand prix
ini_file = "../../VSE/racesim/input/parameters/pars_Spielberg_2019.ini"

#heilmeier's script that runs the race simulator
main_script = "main_racesim.py"

#path to the VSE directory which stores main_racesim.py
vse_dir = "../../VSE"

#list of all drivers who took part in the 2019 austrian grand prix
drivers = ["HAM", "RIC", "MAG", "HUL", "KVY", "RAI", "VET", "BOT", "PER", "GRO", "SAI", "VER", "GIO", "STR", "GAS", "LEC", "NOR", "ALB", "RUS", "KUB"]

def set_vse_type(strategies):
    """
    this function takes a dictionary of driver strategies and updates the ini file
    it reads the whole file in, swaps out the strategy part using some regex magic
    and then writes it all back out to the file
    """
    # open up the ini file and read all the text inside it
    with open(ini_file, 'r') as f:
        content = f.read()
    
    #parse the python dictionary of driver strategies as a json string
    vse_dict_str = json.dumps(strategies)
    
    # use a regular expression to find the old strategy block and replace it with our new one
    new_content = re.sub(r'"vse_type":\s*\{[^}]+\}', '"vse_type": ' + vse_dict_str, content)
    
    # open the file again but this time in write mode so we can overwrite it with the updated text
    with open(ini_file, 'w') as f:
        f.write(new_content)

def get_race_times():
    """
    reads the race times from a csv file given by the simulator, gets the final time for each driver and returns them in a dictionary
    """
    racetime_file = "../../VSE/racesim/output/results/Spielberg_2019_racetimes.csv" #where we save racetime results
    
    # read the csv file into a pandas dataframe
    df = pd.read_csv(racetime_file)
    
    #empty dictionary to hold drivers final times
    times = {}
    
    #go through all drivers
    for d in drivers:
        #check if the driver actually has a column in the results
        if d in df.columns:
            valid_times = df[d].dropna()
            
            # if we have any valid times left grab the very last one (the final cumulative time of finishing the race)
            if not valid_times.empty:
                times[d] = float(valid_times.iloc[-1])
            else:
                #give a 0 if theres no valid times
                times[d] = 0.0
                
    return times

def run_case(case_name, strategies):
    """
    runs a single simulation case (e.g., all drivers follow historical strategy apart from Hamilton who follows the models predicted strategy, or all drivers follow the models predicted strategy)
    """
    print(f"Running Case: {case_name}")
    
    # update the ini file with the strategies for this specific run
    set_vse_type(strategies)
    
    # run the simulator python script using subprocess
    # we tell it to start in the vse directory so it can find all its stuff
    # and we capture the output text so we can read it later
    result = subprocess.run(["python", main_script], cwd=vse_dir, capture_output=True, text=True)
    
    # make an empty dictionary to store the times we scrape from the output
    times = {}
    
    for line in result.stdout.split("\n"):
        if "RESULT:" in line and "Mean Race Time:" in line:
            # split the line into chunks using the colon character
            parts = line.split(":")
            
            # the driver initials are in the second chunk so we clean it up and save it
            initials = parts[1].strip()
            
            # the time is in the fifth chunk we have to clean it up and remove the s at the end
            rt_str = parts[4].strip().replace("s", "")
            
            # turn the time string into a real number and save it in our dictionary
            times[initials] = float(rt_str)
            
    # return the dictionary of times we just scraped
    return times

#all the different simulation scenarios we want to test
cases = [
    ("All Historical", {d: "realstrategy" for d in drivers}),
    ("Historical except HAM (Supervised)", {d: "supervised" if d == "HAM" else "realstrategy" for d in drivers}),
    ("Historical except BOT (Supervised)", {d: "supervised" if d == "BOT" else "realstrategy" for d in drivers}),
    ("All Supervised", {d: "supervised" for d in drivers})
]

#hold the results from all the different cases
results = {}

for name, strats in cases:
    # run the simulation for this scenario and save the times in our results dictionary
    results[name] = run_case(name, strats)
    time.sleep(3)

print("results")
header_row = f"{'Driver'}" + "".join([f" | {name[:20]:<20}" for name, _ in cases])
print(header_row)

# an empty list to store all the rows of text for our table
rows = []

# loop through every driver
for d in drivers:
    # start building the row with the driver initials
    row_str = f"{d:<5}"
    
    # loop through each case to get the time for this driver
    for name, _ in cases:
        # try to get the time from our results dictionary and default to zero if it is missing
        t = results[name].get(d, 0.0)
        # add the time to our row string
        row_str += f" | {t:<20.3f}"
        
    # print the finished row to the screen
    print(row_str)
    
    # add the finished row to our list of rows
    rows.append(row_str)

output_file = "../../runs/simulation_runs/final_results_montecarlo_1000runs.txt"

# open up the output text file so we can write our table into it
with open(output_file, "w") as f:
    f.write(header_row + "\n")
    for r in rows:
        f.write(r + "\n")

# put the original strategy back into the ini file
set_vse_type({d: "supervised" for d in drivers})