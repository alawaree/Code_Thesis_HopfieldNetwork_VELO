import sys
sys.path.append('')
from event_model import event_model as em
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import colorsys


# Solvers
from algorithms.track_following import track_following

def translate(value, left_min, left_max, right_min, right_max):
    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - left_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * right_span)

# Defines a color range for plots later
def color_range(hue_min, hue_max, vmin, vmax):
    return lambda x: colorsys.hsv_to_rgb(translate(x, vmin, vmax, hue_min, hue_max), 1, 1)


total_track_hits = []
# Instantiate algorithm
track_following = track_following()

# Iterate all events
for (dirpath, dirnames, filenames) in os.walk("events"):
  for i, filename in enumerate(filenames):
    # Get an event
    f = open(os.path.realpath(os.path.join(dirpath, filename)))
    json_data = json.loads(f.read())
    event = em.event(json_data)
    f.close()

    # Do track reconstruction TODO: Rather than reconstruct tracks, use the validator to find true tracks
    print("Reconstructing event %i..." % (i))
    tracks = track_following.solve(event)

    # Add the tracks to a total array
    [total_track_hits.append(track.hits) for track in tracks]

# Get an indication of the size of the data
print(len(total_track_hits))

# Used for computed distance to origin
dist = lambda hit: np.sqrt(hit.x**2 + hit.y**2 + hit.z**2)


# Use "min()" for closest (first) hits, use "max()" for furthest (last) hits
# XXX not sure about this here -> who sais that the firt element in the list must be closest to origin... could well be
closest_hits = [min(track, key=dist) for track in total_track_hits]
plt.hist([hit.z for hit in closest_hits])
plt.title("Histogram of z-coordinates for first hits in tracks for 10 events")
plt.show()

plt.hist([dist(hit) for hit in closest_hits])
plt.title("Histogram of distance to origin for first hits in tracks for 10 events")
plt.show()

# Get bounding box for the data to discretize the space for further analysis
min_z = min(closest_hits, key=lambda h: h.z).z
max_z = max(closest_hits, key=lambda h: h.z).z
min_y = min(closest_hits, key=lambda h: h.y).y
max_y = max(closest_hits, key=lambda h: h.y).y
fig, ax = plt.subplots()
plt.scatter([hit.z for hit in closest_hits], [hit.y for hit in closest_hits])

# Choose in how many segments to discretize the space
n = 30
m = 20
zs, step_z = np.linspace(min_z, max_z, n, retstep=True)
ys, step_y = np.linspace(min_y, max_y, m, retstep=True)
rects = np.zeros((n - 1, m - 1))

# Copy the list, as removal of hits is required
closest_hits_copy = closest_hits.copy()
# Find the number of hits in each computed region
for i, z in enumerate(zs[1:]):
    for j, y in enumerate(ys[1:]):
        s = 0
        for hit in closest_hits_copy:
            if hit.z <= z and hit.y <= y:
                s += 1
                closest_hits_copy.remove(hit)
        rects[i][j] = s
print(rects)

# Generate color range from blue (2/3) to red (0) for low to high density regions respectively
# WARNING: This "max()" code below is horrific, but since it's just plots the complexity is fine
get_color = color_range(2/3, 0, 0, max(max(rects, key=lambda x: max(x))))

# Draw the regions on screen colour-coded to number of hits in region
for i, z in enumerate(zs[1:]):
    for j, y in enumerate(ys[1:]):
        rect = plt.Rectangle((z - step_z, y - step_y), step_z, step_y, facecolor=get_color(rects[i][j]), alpha=0.7,
                             edgecolor='black')
        ax.add_patch(rect)
plt.title("Density of first hit in tracks for 10 events. \n"
          "Red = high density. Blue = low density.")
plt.show()

# Create a histogram of the angles from the origin outward. Atan2 function provides 360 degree answer to this.
bins = plt.hist([np.arctan2(hit.y, hit.z) for hit in closest_hits], bins=80)
plt.close()
r = bins[0]
theta = bins[1]
# Instead of plotting a histogram, show in circular fashion the expected directions
for index, val in enumerate(r):
    plt.plot([0, r[index] * np.cos(theta[index])], [0, r[index] * np.sin(theta[index])], color='b')
plt.title("Distribution of direction toward first hits on track for 10 events. \n"
          " Length of line corresponds to number of hits in direction")
plt.show()
