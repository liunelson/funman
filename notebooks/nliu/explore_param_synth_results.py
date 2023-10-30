# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Test latest version of FUNMAN (Functional Model Analysis)

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
# FUNMAN_REST_URL = "http://10.65.18.101:8190/docs"
FUNMAN_REST_URL = 'https://funman.staging.terarium.ai/api'

# %%
# Simple example model (SIR from Model-Representation repo) and request

with open('./results/example_simple/model.json', 'r') as f:
    model = json.load(f)

with open('./results/example_simple/request.json', 'r') as f:
    request = json.load(f)

with open("./results/example_simple/payload.json", "w") as f:
    json.dump({'model': model, 'request': request}, f, indent = 2)

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

        print(f"\tParameter Space:")
        print(f"\t\tNumber of Dimensions: {result['parameter_space']['num_dimensions']}")
        print(f"\t\tNumber of True Boxes: {len(result['parameter_space']['true_boxes'])}")
        if "false_boxes" in result["parameter_space"].keys():
            print(f"\t\tNumber of False Boxes: {len(result['parameter_space']['false_boxes'])}")
    else:
        print(f"Not done")
        print(f"Progress = {result['progress']['progress'] * 100}%")

# %%
# Save result
with open("./results/example_simple/result.json", "w") as f:
    json.dump(result, f, indent = 2)

# %%
# result -> boxes, points, trajectories
def process_funman(result: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    

    df_points = pd.DataFrame()

    df_trajectories = pd.DataFrame()

    # Count objects
    num = {
        k: 0 if k not in result["parameter_space"].keys() else len(result["parameter_space"][k]) 
        for k in ("true_boxes", "false_boxes")
    }
    for k in ("true_points", "false_points"):
        if f"{k.split('_')[0]}_boxes" in result["parameter_space"].keys():
            n = 0
            for box in result["parameter_space"][f"{k.split('_')[0]}_boxes"]:
                n += len(box["points"])
            num[k] = n
    print(f"Number of boxes = {num['true_boxes']} True + {num['false_boxes']} False = {num['true_boxes'] + num['false_boxes']}")
    print(f"Number of points = {num['true_points']} True + {num['false_points']} False = {num['true_points'] + num['false_points']}")

    # List of states and parameters
    states = [s["id"] for s in result["model"]["petrinet"]["model"]["states"]]
    params = [p["id"] for p in result["model"]["petrinet"]["semantics"]["ode"]["parameters"]]


    # Give IDs to all boxes
    for i, box in enumerate(result["parameter_space"]["true_boxes"] + result["parameter_space"]["false_boxes"]):
        box["id"] = f"box{i}"

    # Give IDs to all points
    i = 0
    for box in (result["parameter_space"]["true_boxes"] + result["parameter_space"]["false_boxes"]):
        for point in box["points"]:
            point["id"] = f"point{i}"
            i += 1

    # DF for boxes
    boxes = []
    for k in ("true_boxes", "false_boxes"):
        if k in result["parameter_space"].keys():
            for box in result["parameter_space"][k]:
                boxes.append([box["id"], box["label"], box["bounds"]["timestep"]["ub"]] + [(box["bounds"][p]["lb"], box["bounds"][p]["ub"]) for p in params])
    df_boxes = pd.DataFrame(
        data = boxes,
        columns = ["id", "label", "timestep"] + params, 
        dtype = None
    )

    # DF for points
    points = []
    for k in ("true_boxes", "false_boxes"):
        if k in result["parameter_space"].keys():
            for box in result["parameter_space"][k]:
                for point in box["points"]:
                    points.append([point["id"], point["label"], box["id"]] + [point["values"][p] for p in params])
    df_points = pd.DataFrame(
        data = points,
        columns = ["id", "label", "box_id"] + params
    )

    # DF for trajectories
    trajs = []
    for k in ("true_boxes", "false_boxes"):
        if k in result["parameter_space"].keys():
            for box in result["parameter_space"][k]:
                for point in box["points"]:
                    values = {l: v for l, v in point["values"].items() if (l not in params) & (l != "timestep") & (l.split("_")[0] != "assume")}
                    timestep_ids = sorted(list(set([l.rpartition("_")[-1] for l in values.keys()])))
                    for t in timestep_ids:
                        trajs.extend([[box["id"], point["id"], t, values[f"timer_t_{t}"]] + [values[f"{s}_{t}"] for s in states]])

    df_trajs = pd.DataFrame(
        data = trajs,
        columns = ["box_id", "point_id", "timestep_id", "time"] + states
    )

    return (df_boxes, df_points, df_trajs)

# %%
(df_boxes, df_points, df_trajs) = process_funman(result)

# %%
fig, axes = plt.subplots(1, 2, figsize = (12, 6))
colors = mpl.cm.tab10(plt.Normalize(0, 10)(range(10)))
map_colors = {"true": colors[0], "false": colors[1]}

# Draw boxes
b = "box27"
t = 7.0
i = df_boxes["timestep"] == t
for __, box in df_boxes[i].iterrows():
    p = mpl.patches.Rectangle(
        (box["beta"][0], box["gamma"][0]), 
        box["beta"][1] - box["beta"][0], 
        box["gamma"][1] - box["gamma"][0],
        facecolor = map_colors[box["label"]],
        edgecolor = map_colors[box["label"]],
        alpha = 0.3
    )
    if box["id"] == b:
        __ = plt.setp(p, alpha = 0.6)

    __ = axes[0].add_patch(p)

# Draw points
j = df_points["box_id"].isin(df_boxes[i]["id"])
for __, point in df_points[j].iterrows():
    __ = axes[0].scatter(point["beta"], point["gamma"], color = map_colors[point["label"]], marker = '.')

__ = plt.setp(
    axes[0], 
    title = f"All Boxes, Coloured by True/False ({b} Selected = Darker)",
    xlabel = "beta",
    ylabel = "gamma",
    xlim = (result["request"]["parameters"][0]["interval"]["lb"], result["request"]["parameters"][0]["interval"]["ub"]),
    ylim = (result["request"]["parameters"][1]["interval"]["lb"], result["request"]["parameters"][1]["interval"]["ub"]),
)

# Draw trajectory
k = df_trajs["box_id"] == b
time = df_trajs[k]["time"]
for s in list(df_trajs.columns)[4:]:
    __ = axes[1].plot(time, df_trajs[k][s], label = s)

# Add constraints
result["request"]["constraints"][0]
p = mpl.patches.Rectangle(
    (result["request"]["constraints"][0]["timepoints"]["lb"], result["request"]["constraints"][0]["interval"]["lb"]),
    result["request"]["constraints"][0]["timepoints"]["ub"] - result["request"]["constraints"][0]["timepoints"]["lb"],
    result["request"]["constraints"][0]["interval"]["ub"] - result["request"]["constraints"][0]["interval"]["lb"],
    facecolor = "w",
    edgecolor = "k",
    linestyle = "--",
    alpha = 0.5,
    label = 'Constraints'
)
__ = axes[1].add_patch(p)

__ = plt.setp(
    axes[1],
    title = f"All trajectories of Box ID = {b}",
    xlabel = "Timestep (days)",
    ylabel = "State Variable Values"
)
__ = axes[1].legend()

# %%
