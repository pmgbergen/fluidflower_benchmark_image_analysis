"""
Collect all results for C1-C5 and store in a single file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from src.io import read_paths_from_user_data

# Fetch data
length_fingers = {}
for run in ["c1", "c2", "c3", "c4", "c5"]:
    # Read user-defined paths to images etc.
    _, _, _, results = read_paths_from_user_data("../data.json", run)
    length_fingers[run] = np.genfromtxt(
        results / Path("sparse_data") / Path("length_fingers.csv"), delimiter=","
    )

print(length_fingers["c1"].shape)
print(length_fingers["c2"].shape)
print(length_fingers["c3"].shape)
print(length_fingers["c4"].shape)
print(length_fingers["c5"].shape)

# Check relative times
# print((length_fingers["c1"][:,0] - length_fingers["c2"][:,0]) / length_fingers["c1"][:,0])
# print((length_fingers["c1"][:,0] - length_fingers["c3"][:,0]) / length_fingers["c1"][:,0])
# print((length_fingers["c1"][:,0] - length_fingers["c4"][:,0]) / length_fingers["c1"][:,0])
# print((length_fingers["c1"][72:,0] - length_fingers["c5"][68:,0]) / length_fingers["c1"][72:,0])

plt.plot(length_fingers["c1"][:, 1])
plt.show()
# Remove indices 68, 69, 70, 71 from c1, c2, c3, c4 as they are missing in c5.
mask = np.ones(len(length_fingers["c1"]), dtype=bool)
mask[68:72] = 0
length_fingers["c1"] = length_fingers["c1"][mask, :]
length_fingers["c2"] = length_fingers["c2"][mask, :]
length_fingers["c3"] = length_fingers["c3"][mask, :]
length_fingers["c4"] = length_fingers["c4"][mask, :]
plt.plot(length_fingers["c1"][:, 1])
plt.show()

# print((length_fingers["c1"][:,0] - length_fingers["c2"][:,0]) / length_fingers["c1"][:,0])
# print((length_fingers["c1"][:,0] - length_fingers["c3"][:,0]) / length_fingers["c1"][:,0])
# print((length_fingers["c1"][:,0] - length_fingers["c4"][:,0]) / length_fingers["c1"][:,0])
# print((length_fingers["c1"][:,0] - length_fingers["c5"][:,0]) / length_fingers["c1"][:,0])

# Combine results in a single csv file
length_fingers = np.zeros((len(length_fingers["c1"]), 6), dtype=float)
# Time
length_fingers[:, 0] = length_fingers["c1"][:, 0]
# C1-C5
length_fingers[:, 1] = length_fingers["c1"][:, 1]
length_fingers[:, 2] = length_fingers["c2"][:, 1]
length_fingers[:, 3] = length_fingers["c3"][:, 1]
length_fingers[:, 4] = length_fingers["c4"][:, 1]
length_fingers[:, 5] = length_fingers["c5"][:, 1]

# Store to file
finger_header = f"Time in hours, length of finger tips in box C for C1, C2, C3, C4, C5"
fmt = "%f", "%f", "%f", "%f", "%f", "%f"
np.savetxt(
    "length_fingers.csv", length_fingers, fmt=fmt, delimiter=",", header=finger_header
)

plt.figure("length of fingers in C - 5 days")
plt.plot(length_fingers[:, 0], length_fingers[:, 1], label="c1")
plt.plot(length_fingers[:, 0], length_fingers[:, 2], label="c2")
plt.plot(length_fingers[:, 0], length_fingers[:, 3], label="c3")
plt.plot(length_fingers[:, 0], length_fingers[:, 4], label="c4")
plt.plot(length_fingers[:, 0], length_fingers[:, 5], label="c5")
plt.legend()

one_day = np.argmin(np.absolute(length_fingers[:, 0] - 24))
print(one_day)
plt.figure("length of fingers in C - 1 day")
plt.plot(length_fingers[:one_day, 0], length_fingers[:one_day, 1], label="c1")
plt.plot(length_fingers[:one_day, 0], length_fingers[:one_day, 2], label="c2")
plt.plot(length_fingers[:one_day, 0], length_fingers[:one_day, 3], label="c3")
plt.plot(length_fingers[:one_day, 0], length_fingers[:one_day, 4], label="c4")
plt.plot(length_fingers[:one_day, 0], length_fingers[:one_day, 5], label="c5")
plt.legend()

ten_hours = np.argmin(np.absolute(length_fingers[:, 0] - 10))
print(ten_hours)
plt.figure("length of fingers in C - 10 hours")
plt.plot(length_fingers[:ten_hours, 0], length_fingers[:ten_hours, 1], label="c1")
plt.plot(length_fingers[:ten_hours, 0], length_fingers[:ten_hours, 2], label="c2")
plt.plot(length_fingers[:ten_hours, 0], length_fingers[:ten_hours, 3], label="c3")
plt.plot(length_fingers[:ten_hours, 0], length_fingers[:ten_hours, 4], label="c4")
plt.plot(length_fingers[:ten_hours, 0], length_fingers[:ten_hours, 5], label="c5")
plt.legend()

four_hours = np.argmin(np.absolute(length_fingers[:, 0] - 4))
seventeen_hours = np.argmin(np.absolute(length_fingers[:, 0] - 17))
print(ten_hours)
plt.figure("length of fingers in C - 4-17 hours")
plt.plot(
    length_fingers[four_hours:seventeen_hours, 0],
    length_fingers[four_hours:seventeen_hours, 1],
    label="c1",
)
plt.plot(
    length_fingers[four_hours:seventeen_hours, 0],
    length_fingers[four_hours:seventeen_hours, 2],
    label="c2",
)
plt.plot(
    length_fingers[four_hours:seventeen_hours, 0],
    length_fingers[four_hours:seventeen_hours, 3],
    label="c3",
)
plt.plot(
    length_fingers[four_hours:seventeen_hours, 0],
    length_fingers[four_hours:seventeen_hours, 4],
    label="c4",
)
plt.plot(
    length_fingers[four_hours:seventeen_hours, 0],
    length_fingers[four_hours:seventeen_hours, 5],
    label="c5",
)
plt.legend()

plt.show()
