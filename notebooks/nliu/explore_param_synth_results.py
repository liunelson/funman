# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Test FUNMAN (Functional Model Analysis)
# 
# 1. Test FUNMAN API
# 2. Explore example output from SIFT team
# 3. Try regenerating the 12-month milestone scenarios

#
# Script for generating the scenarios:
# 

# %%
import os
import json
import requests
import time
from typing import NoReturn, Optional, Any
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
FUNMAN_REST_URL = 'https://funman.staging.terarium.ai/api'

# %%
# # 1. Make Test Query

# %%
# Example model
with open('../../resources/amr/petrinet/amr-examples/sir_typed.json', 'r') as f:
    model = json.load(f)

# Example request
with open('../../resources/amr/petrinet/amr-examples/sir_request1.json', 'r') as f:
    request = json.load(f)

# %%
# request = {
#     "query": {
#         "queries": [
#             {
#                 "variable": "I", 
#                 "lb": 0.0,
#             },
#             {
#                 "variable": "I", 
#                 "ub": 0.3
#             }
#         ]
#     }
#     ,
#     "parameters": [
#         {
#             "name": "beta", 
#             "lb": 0.01,
#             "ub": 1.0
#         },
#         {
#             "name": "gamma",
#             "lb": 0.01,
#             "ub": 1.0
#         },
#         {
#             "name": "S0",
#             "lb": 0.9,
#             "ub": 1.0
#         },
#         {
#             "name": "I0",
#             "lb": 0.0,
#             "ub": 0.1
#         },
#         {
#             "name": "R0",
#             "lb": 0.0,
#             "ub": 0.1
#         }
#     ],
#     "structure_parameters": [
#         {
#             "name": "step_size",
#             "lb": 1.0,
#             "ub": 1.0,
#             "label": "all"
#         },
#         {
#             "name": "num_steps",
#             "lb": 10.0,
#             "ub": 10.0,
#             "label": "all"
#         }
#     ],
#     "config": {
#         "solver": "dreal",
#         "dreal_mcts": True,
#         "tolerance": 1e-8
#     }
# }


# %%
# Make FUNMAN query

data = json.dumps({
    'model': model,
    'request': request
}, indent = 2)

url = f'{FUNMAN_REST_URL}/queries'
res = requests.post(url = url, data = data)
print(f'{res.url}: Code {res.status_code}\n\t\"{res.text}\"')
if res.status_code == 200:
    query_id = res.json()['id']

# %%
# Pause to wait for FUNMAN to finish running
time.sleep(3)

# %%
# Retrieve FUNMAN result
res = requests.get(url = f"{FUNMAN_REST_URL}/queries/{query_id}")
print(f"{res.url}: Code {res.status_code}\n\t\"{res.text}\"")
if res.status_code == 200:
    result = res.json()
    if "done" in result.keys():
        __ = [print(f"\t{k}: {result[k]}") for k in ('id', 'done')]
        __ = print(f"\tParameter Space:\n\t\tNumber of True Boxes: {len(result['parameter_space']['true_boxes'])}\n\t\tNumber of True Points: {len(result['parameter_space']['true_points'])}")
    else:
        print(f"Not done")

# %%
def plot_funman_result(result: dict, num_parameters_show: Optional[int] = None, point_id: Optional[int] = None) -> NoReturn:

    query_id = result["id"]

    # Model parameters and "structure" parameters
    parameters = list(set([p["id"] for p in result["model"]["petrinet"]["semantics"]["ode"]["parameters"]]))
    struct_parameters = list(set([p["name"] for p in result["request"]["structure_parameters"]]))
    num_parameters = len(parameters)
    
    # How many parameters to show
    if (num_parameters_show == None) or (num_parameters_show > num_parameters):
        num_parameters_show = num_parameters
    
    # Parameter request ranges (lower and upper bounds)
    map_ranges = {p: {"lb": None, "ub": None, "value": None} for p in parameters}
    
    # Check inside request
    for p in result["request"]["parameters"]:
        for k in ["lb", "ub", "value"]:
            if k in p.keys():
                map_ranges[p["name"]][k] = p[k]

    # Check inside model configuration
    for p in result["model"]["petrinet"]["semantics"]["ode"]["parameters"]:
        for k in ["lb", "ub", "value"]:
            if k in p.keys():
                map_ranges[p["id"]][k] = p[k]

    # If parameter ranges are missing, assume +/- 10%
    PARAM_RANGE_TOL = 0.10
    for p in map_ranges.keys():
        if map_ranges[p]["value"] != None:
            if map_ranges[p]["lb"] == None:
                map_ranges[p]["lb"] = (1.0 - PARAM_RANGE_TOL) * map_ranges[p]["value"]
            if map_ranges[p]["ub"] == None:
                map_ranges[p]["ub"] = (1.0 + PARAM_RANGE_TOL) * map_ranges[p]["value"]
        
        if (map_ranges[p]["lb"] == map_ranges[p]["ub"]) and (map_ranges[p]["lb"] != None):
            map_ranges[p]["value"] = map_ranges[p]["lb"]
            map_ranges[p]["lb"] = (1.0 - PARAM_RANGE_TOL) * map_ranges[p]["value"]
            map_ranges[p]["ub"] = (1.0 + PARAM_RANGE_TOL) * map_ranges[p]["value"]

    # Number of points
    num_true_points = len(result["parameter_space"]["true_points"])
    num_false_points = len(result["parameter_space"]["false_points"])
    num_points = num_true_points + num_false_points
    print(f"Number of points = {num_points}")
    print(f"Number of true points = {num_true_points}")
    print(f"Number of false points = {num_false_points}")

    # Color map
    colors = mpl.cm.tab10(plt.Normalize(0, 10)(range(10)))
    map_colors = {"true": colors[0], "false": colors[1]}

    # Figure and axes
    fig, axes = plt.subplots(nrows = num_parameters_show, ncols = num_parameters_show, figsize = (12, 12))
    fig.suptitle(f"{result['model']['petrinet']['name']}")

    for (i, j) in itertools.product(tqdm(range(num_parameters_show)), repeat = 2):

            if i == j:

                # Total distribution
                r = [map_ranges[parameters[i]]["lb"], map_ranges[parameters[i]]["ub"]]
                c = [map_colors["true"], map_colors["false"]]
                x = [[point["values"][parameters[i]] for point in result["parameter_space"][k]] for k in ["true_points", "false_points"]]
                __ = axes[i, j].hist(x, bins = 11, range = r, color = c, alpha = 1.0, stacked = True, align = "mid", density = False)

            elif j < i:
                
                # True and false boxes
                boxes = []
                for k in ("true_boxes", "false_boxes"):

                    c = map_colors[k.split("_")[0]]

                    for box in result["parameter_space"][k]:
    
                        x = box["bounds"][parameters[j]]["lb"]
                        y = box["bounds"][parameters[i]]["lb"]
                        w = box["bounds"][parameters[j]]["ub"] - box["bounds"][parameters[j]]["lb"]
                        h = box["bounds"][parameters[i]]["ub"] - box["bounds"][parameters[i]]["lb"]

                        if (w > 0) and (h > 0):
                            boxes.append(mpl.patches.Rectangle((x, y), w, h, facecolor = c, edgecolor = c, alpha = 0.2))
                        elif (w + h) > 0:
                            __ = axes[i, j].plot(
                                [x, box["bounds"][parameters[j]]["ub"]], 
                                [y, box["bounds"][parameters[i]]["ub"]], 
                                color = c,
                                alpha = 0.2,
                                marker = None
                            )
                        else:
                            pass

                    pc = mpl.collections.PatchCollection(boxes, edgecolor = None)
                    axes[i, j].add_collection(pc)

                # True points
                for k in ("true_points", "false_points"):

                    c = map_colors[k.split("_")[0]]

                    for point in result["parameter_space"][k]:
                        x = point["values"][parameters[j]]
                        y = point["values"][parameters[i]]
                        __ = axes[i, j].plot(x, y, '.', color = c, alpha = 0.2, linestyle = None)
                
                # Axis ranges
                if map_ranges[parameters[j]]["lb"] < map_ranges[parameters[j]]["ub"]:
                    __ = plt.setp(axes[i, j], xlim = (map_ranges[parameters[j]]["lb"], map_ranges[parameters[j]]["ub"]))
                if map_ranges[parameters[i]]["lb"] < map_ranges[parameters[i]]["ub"]:
                    __ = plt.setp(axes[i, j], ylim = (map_ranges[parameters[i]]["lb"], map_ranges[parameters[i]]["ub"]))
            
            else:
                axes[i, j].remove()

            axes[i, j].tick_params(axis = 'x', labelrotation = 45)
            if (j == 0) and (i > 0):
                __ = plt.setp(axes[i, j], ylabel = parameters[i])
            else:
                axes[i, j].tick_params(axis = 'y', labelleft = False)

            if (i == (num_parameters_show - 1)):
                __ = plt.setp(axes[i, j], xlabel = parameters[j])
            else:
                axes[i, j].tick_params(axis = 'x', labelbottom = False)

            if (i == j):
                axes[i, j].tick_params(axis = 'y', left = False, right = True, labelright = True)
                # __ = plt.setp(axes[i, j], ylabel = f"H")
                axes[i, j].yaxis.set_label_position("right")

# %%
# plot_funman_result(result)

# %%[markdown]
# # 2. Explore Example Output

# %%
# Check all available examples
# Find one with nonzero numer of time points and both true and false points/boxes

for query_id in os.listdir("./results/milestone_12month/"):

    p = f"./results/milestone_12month/{query_id}"

    with open(p, 'r') as f:
        result = json.load(f)

        print(f'Query ID:\t\t{query_id}')
        print(f'Done:\t\t\t{result["done"]}')
        print(f'Model:\t\t\t{result["model"]["petrinet"]["name"]}')
        print(f'Number of parameters:\t{len(result["model"]["petrinet"]["semantics"]["ode"]["parameters"])}')
        
        m = {p["name"]: p for p in result["request"]["structure_parameters"]}
        print(f'Number of time points:\t{m["num_steps"]["ub"]:.0f}')

        __ = [print(f'Number of {" ".join(k.split("_"))}:\t{len(result["parameter_space"][k])}') for k in ("true_points", "true_boxes", "false_points", "false_boxes")]
        print("\n")

# %%
# Query ID:		863e1aa6-ac9b-4697-b8f9-833fef1e4f31.json
# Done:			True
# Model:			Scenario 2a
# Number of parameters:	15
# Number of time points:	2
# Number of true points:	2
# Number of true boxes:	2
# Number of false points:	0
# Number of false boxes:	0


# Query ID:		3aaa15e5-15fc-43f5-ba51-acbc75fb2181.json
# Done:			False
# Model:			Evaluation Scenario 1. Part 1 (ii) Masking type 3
# Number of parameters:	21
# Number of time points:	50
# Number of true points:	3661
# Number of true boxes:	169
# Number of false points:	3492
# Number of false boxes:	285


# Query ID:		4b70df7e-cb11-4215-9b22-94edb045a3fd.json
# Done:			False
# Model:			Evaluation Scenario 1. Part 1 (ii) Masking type 1
# Number of parameters:	13
# Number of time points:	200
# Number of true points:	2224
# Number of true boxes:	368
# Number of false points:	1856
# Number of false boxes:	574


# Query ID:		81bb3877-35a0-4dc5-8bd9-93e6d4d31c07.json
# Done:			True
# Model:			Evaluation Scenario 1. Part 1 (ii) Masking type 3
# Number of parameters:	21
# Number of time points:	150
# Number of true points:	1
# Number of true boxes:	0
# Number of false points:	0
# Number of false boxes:	0


# Query ID:		727d1c44-9c0c-4a89-a059-6cb53f4fa7fd.json
# Done:			False
# Model:			Evaluation Scenario 1. Part 1 (ii) Masking type 1
# Number of parameters:	13
# Number of time points:	200
# Number of true points:	542
# Number of true boxes:	96
# Number of false points:	446
# Number of false boxes:	95


# Query ID:		a88865c8-dedd-42ee-9ca2-ae311be02d70.json
# Done:			True
# Model:			Evaluation Scenario 1 Base model
# Number of parameters:	11
# Number of time points:	200
# Number of true points:	766
# Number of true boxes:	387
# Number of false points:	379
# Number of false boxes:	349


# Query ID:		c1134f9d-a75c-4a46-9df5-349868b6ff56.json
# Done:			False
# Model:			Evaluation Scenario 1. Part 1 (ii) Masking type 2
# Number of parameters:	15
# Number of time points:	100
# Number of true points:	101
# Number of true boxes:	101
# Number of false points:	0
# Number of false boxes:	0


# Query ID:		e9499da0-ef8c-4d58-9386-95a15aad918a.json
# Done:			True
# Model:			Evaluation Scenario 1. Part 1 (ii) Masking type 3
# Number of parameters:	21
# Number of time points:	50
# Number of true points:	1
# Number of true boxes:	0
# Number of false points:	0
# Number of false boxes:	0


# Query ID:		e4917013-2254-4a79-b653-51c3da94522b.json
# Done:			False
# Model:			Evaluation Scenario 1. Part 1 (ii) Masking type 3
# Number of parameters:	21
# Number of time points:	50
# Number of true points:	15
# Number of true boxes:	15
# Number of false points:	0
# Number of false boxes:	0


# Query ID:		f339aafc-749a-417c-afd6-fad1429e43ed.json
# Done:			True
# Model:			Evaluation Scenario 1. Part 1 (ii) Masking type 3
# Number of parameters:	21
# Number of time points:	200
# Number of true points:	1
# Number of true boxes:	0
# Number of false points:	0
# Number of false boxes:	0

# %%
# query_id = "3aaa15e5-15fc-43f5-ba51-acbc75fb2181.json"
# query_id = "4b70df7e-cb11-4215-9b22-94edb045a3fd.json"
query_id = "727d1c44-9c0c-4a89-a059-6cb53f4fa7fd.json"
# query_id = "a88865c8-dedd-42ee-9ca2-ae311be02d70.json"
# query_id = "f339aafc-749a-417c-afd6-fad1429e43ed.json"
p = f"./results/milestone_12month/{query_id}"

with open(p, 'r') as f:
    result = json.load(f)

    print(f'Query ID:\t\t{query_id}')
    print(f'Done:\t\t\t{result["done"]}')
    print(f'Model:\t\t\t{result["model"]["petrinet"]["name"]}')
    print(f'Number of parameters:\t{len(result["model"]["petrinet"]["semantics"]["ode"]["parameters"])}')
    
    m = {p["name"]: p for p in result["request"]["structure_parameters"]}
    print(f'Number of time points:\t{m["num_steps"]["ub"]:.0f}')

    __ = [print(f'Number of {" ".join(k.split("_"))}:\t{len(result["parameter_space"][k])}') for k in ("true_points", "true_boxes", "false_points", "false_boxes")]
    print("\n")


# %%
plot_funman_result(result, num_parameters_show = None)
fig = plt.clf()
fig.savefig(f"./results/milestone_12month/{query_id}.png")

# %%
# Collate the trajectories into a dataframe
def get_FUNMAN_trajectories(result: dict) -> pd.DataFrame:

    result_df = pd.DataFrame()

    num_true_points = len(result["parameter_space"]["true_points"])
    num_false_points = len(result["parameter_space"]["false_points"])
    num_points = num_true_points + num_false_points
    print(f"Number of points = {num_points}")
    print(f"Number of true points = {num_true_points}")
    print(f"Number of false points = {num_false_points}")

    if num_points > 0:

        point_ids = [i for i in range(num_points)]

        # Get list of parameters
        parameters = set([p["id"] for p in result["model"]["petrinet"]["semantics"]["ode"]["parameters"]])
        struct_parameters = set([p["name"] for p in result["request"]["structure_parameters"]])

        # Collect trajectories per true point
        for i, point in enumerate(result["parameter_space"]["true_points"] + result["parameter_space"]["false_points"]):

            # Get list of state variables and time points
            x = set(list(point["values"].keys())) - parameters - struct_parameters
            x = [s for s in x if s[:12] != "assume_query"] # remove "assume_query_X" keys

            if len(x) > 0:

                state_vars, time_points = np.array([s.rsplit('_', 1) for s in x]).T
                state_vars = sorted(list(set(state_vars) - {"timer_t"}))
                time_points = sorted(list(set([int(t) for t in time_points])))

                # Patch weird bug when time resets to 1.0 
                # when more time points/steps are generated than `num_steps`
                num_steps = int(point["values"]["num_steps"]) + 1
                if len(time_points) > num_steps:
                    time_points = time_points[:num_steps]
                
                df = pd.DataFrame(
                    data = np.array([[i, point["label"] == 'true', t, point["values"][f"timer_t_{str(t)}"]] + [point["values"][f"{v}_{str(t)}"] for v in state_vars] for t in time_points]),
                    columns = ["point_id", "label", "time_point_id", "time"] + state_vars,
                    dtype = None
                )

                result_df = pd.concat([result_df, df], ignore_index = True)

            else:
                print(f"No time points")

    result_df.attrs = {
        "model_name": result["model"]["petrinet"]["name"]
    }

    # Set types
    for c in result_df.columns:
        if c in ["point_id", "time_point_id"]:
            result_df[c] = result_df[c].astype("int")
        elif c in ["label"]:
            result_df[c] = result_df[c].astype("bool")
        else:
            result_df[c] = result_df[c].astype("float")

    return result_df

# %%
result_df = get_FUNMAN_trajectories(result)
result_df

# %%
result_df.describe()

# %%
# Plot trajectory of a particular true/false point

point_id = 541

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
for v in result_df.columns[4:]:
    k = result_df["point_id"] == point_id
    t = result_df[k]["time"]
    x = result_df[k][v]
    __ = ax.plot(t, x, label = f"{v}")

__ = plt.setp(ax, xlabel = "Time", ylabel = "State Variable Values", title = f'{result_df.attrs["model_name"]} - {result_df[k]["label"].iloc[0]} Point')
__ = ax.legend()
fig.savefig("./results/milestone_12month/all_vars_point_541.png")

# %%
# Plot all trajectories for a given  state variable

state_var = "I"

fig, ax = plt.subplots(1, 1, figsize = (8, 6))

colors = mpl.cm.tab10(plt.Normalize(0, 10)(range(10)))
map_colors = {"true": colors[0], "false": colors[1]}
num_points = max(result_df["point_id"])

for point_id in range(num_points):

    k = result_df["point_id"] == point_id
    t = result_df[k]["time"]
    x = result_df[k][state_var]
    c = map_colors[str(result_df[k]["label"].iloc[0]).lower()]
    __ = ax.plot(t, x, color = c, alpha = 0.1, label = f"{point_id}")

__ = plt.setp(ax, xlabel = "Time", ylabel = f"{state_var} Values", title = f'{result_df.attrs["model_name"]} - All Points')

fig.savefig("./results/milestone_12month/var_I_all_points.png")

# %%

