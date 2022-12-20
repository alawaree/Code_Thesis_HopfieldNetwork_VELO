#!/usr/bin/python3
import matplotlib as mpl
# mpl.use('Agg')

from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import math

# Beautiful colors
default_color = "#478DCB"
grey_color = "#D0D0D0"
colors = ["#CF3D1E", "#F15623", "#F68B1F", "#FFC60B", "#DFCE21",
  "#BCD631", "#95C93D", "#48B85C", "#00833D", "#00B48D", 
  "#60C4B1", "#27C4F4", "#3E67B1", "#4251A3", "#59449B", 
  "#6E3F7C", "#6A246D", "#8A4873", "#EB0080", "#EF58A0", "#C05A89"]

# Some default parameters for the figure
scale = 4
plotscale = 1.5

# # Dashed line for modules
# plt.plot(
#   [a for a in range(1, 256)],
#   [a for a in range(1, 256)],
#   '--',
#   color=grey_color
# )

ntox = {0:'X', 1:'Y', 2:'Z'}

# def hit_phi(hit):
#   if (hit.module_number % 2) == 0:
#     phi = math.atan2(hit.y, hit.x)
#     less_than_zero = phi < 0
#     return phi + less_than_zero * 2 * math.pi
#   return math.atan2(hit.y, hit.x)


def hit_phi(x, y):
    return math.atan2(y, x)


def hit_r(x, y):
    return math.sqrt(x**2 + y**2)


def print_event_2d_2phi(event, tracks, phix=[0,1], phiy=[1,2], filename="doublephi", save_to_file=False):
  fig = plt.figure(figsize=(16*plotscale, 9*plotscale))
  ax = plt.axes()

  count = 0
  for t in [t for t in tracks if len(t.hits)>=3]:
    count+=1
    plt.plot(
      [hit_phi(h[phix[0]], h[phix[1]]) for h in t.hits],
      [hit_phi(h[phiy[0]], h[phiy[1]]) for h in t.hits],
      color=colors[count % len(colors)],
      alpha=0.4,
      linewidth=1
    )
    plt.scatter(
      [hit_phi(h[phix[0]], h[phix[1]]) for h in t.hits],
      [hit_phi(h[phiy[0]], h[phiy[1]]) for h in t.hits],
      color=colors[count % len(colors)],
      s=2*scale
    )

  plt.tick_params(axis='both', which='major', labelsize=4*scale)
  plt.xlabel(f'{ntox[phix[0]]}-{ntox[phix[1]]}', fontdict={'fontsize': 4*scale})
  plt.ylabel(f'{ntox[phiy[0]]}-{ntox[phiy[1]]}', fontdict={'fontsize': 4*scale}, rotation='horizontal')
  if save_to_file:
    plt.savefig(filename + ".png", bbox_inches='tight', pad_inches=0.2)
    plt.close()
  else:
    plt.show()


def print_event_2d_2phi_r(event, tracks, x=2, phix=0, phiy=1, filename="double_phi_with_r", save_to_file=False):
    fig = plt.figure(figsize=(16*plotscale, 9*plotscale))
    ax = plt.axes()

    if len(tracks) > 0:
        count = 0
        for t in [t for t in tracks if len(t.hits)  >= 3]:
            plt.plot(
              [hit_phi(hit_r(h[phix], h[phiy]), h[x]) for h in t.hits],
              [hit_phi(h[phix], h[phiy]) for h in t.hits],
              color=colors[count % len(colors)],
              alpha=0.4,
              linewidth=1
            )
            plt.scatter(
              [hit_phi(hit_r(h[phix], h[phiy]), h[x]) for h in t.hits],
              [hit_phi(h[phix], h[phiy]) for h in t.hits],
              color=colors[count % len(colors)],
              s=2 * scale
            )
            count += 1

    else:
        plt.scatter(
          [hit_phi(hit_r(h[phix], h[phiy]), h[x]) for h in event.hits],
          [hit_phi(h[phix], h[phiy]) for h in event.hits],
          color=default_color,
          s=2 * scale
        )

    plt.title(filename)
    plt.tick_params(axis='both', which='major', labelsize=4*scale)
    plt.xlabel(f'angle(r{ntox[x]})', fontdict={'fontsize': 4*scale})
    angle = ""
    if phix == 0 and phiy == 1:
        angle = "φ"
    if phix == 1 and phiy == 2:
        angle = "θ"
    if phix == 2 and phiy == 0:
        angle = "Ψ"
    plt.ylabel(angle, fontdict={'fontsize': 4*scale}, rotation='horizontal')

    if save_to_file:
        plt.savefig(filename + ".png", bbox_inches='tight', pad_inches=0.2)
        plt.close()
    else:
        plt.show()

def print_event_2d_phi_r(event, tracks, phix=0, phiy=1, filename="event_phi_r", save_to_file=False):
  fig = plt.figure(figsize=(16*plotscale, 9*plotscale))
  ax = plt.axes()
  if len(tracks) > 0:
    count = 0
    for t in [t for t in tracks if len(t.hits)>=3]:
      plt.plot(
        [hit_r(h[phix], h[phiy]) for h in t.hits],
        [hit_phi(h[phix], h[phiy]) for h in t.hits],
        color=colors[count % len(colors)],
        alpha=0.4,
        linewidth=1
      )
      plt.scatter(
        [hit_r(h[phix], h[phiy]) for h in t.hits],
        [hit_phi(h[phix], h[phiy]) for h in t.hits],
        color=colors[count % len(colors)],
        s=2*scale
      )
      count+=1

  else:
    plt.scatter(
      [hit_r(h[phix], h[phiy]) for h in event.hits],
      [hit_phi(h[phix], h[phiy]) for h in event.hits],
      color=default_color,
      s=2*scale
    )

  plt.title(filename)
  plt.tick_params(axis='both', which='major', labelsize=4*scale)
  plt.xlabel("r", fontdict={'fontsize': 4*scale})
  angle = ""
  if phix == 0 and phiy == 1:
    angle = "φ"
  if phix == 1 and phiy == 2:
    angle = "θ"
  if phix == 2 and phiy == 0:
    angle = "Ψ"
  plt.ylabel(angle, fontdict={'fontsize': 4*scale}, rotation='horizontal')

  if save_to_file:
    plt.savefig(filename + ".png", bbox_inches='tight', pad_inches=0.2)
    plt.close()
  else:
    plt.show()


def print_event_2d_phi(event, tracks, x=2, track_color=0, phix=0, phiy=1, filename="event_phi", save_to_file=False):
  fig = plt.figure(figsize=(16*plotscale, 9*plotscale))
  ax = plt.axes()

  # Limits of the modules
  limits = [(-math.pi, math.pi), (-math.pi, math.pi)]
  shift = 0.4
  for s in event.modules[::2]:
    plt.plot(
      [s.z+shift, s.z+shift],
      [limits[0][0], limits[0][1]],
      color=grey_color,
      alpha=0.4,
      linewidth=4
    )

  for s in event.modules[1::2]:
    plt.plot(
      [s.z+shift, s.z+shift],
      [limits[1][0], limits[1][1]],
      color=grey_color,
      alpha=0.4,
      linewidth=4
    )

  if len(tracks) > 0:
    count = 0
    for t in [t for t in tracks if len(t.hits)>=3]:
      plt.plot(
        [h[x] for h in t.hits],
        [hit_phi(h[phix], h[phiy]) for h in t.hits],
        color=colors[count % len(colors)],
        alpha=0.4,
        linewidth=1
      )
      plt.scatter(
        [h[x] for h in t.hits],
        [hit_phi(h[phix], h[phiy]) for h in t.hits],
        color=colors[count % len(colors)],
        s=2*scale
      )
      count+=1

  else:
    plt.scatter(
      [h[x] for h in event.hits],
      [hit_phi(h[phix], h[phiy]) for h in event.hits],
      color=default_color,
      s=2*scale
    )

  plt.title(filename)
  plt.tick_params(axis='both', which='major', labelsize=4*scale)
  plt.xlabel(f'{ntox[x]}', fontdict={'fontsize': 4*scale})   # TODO: label "Z"?
  angle = ""
  if phix == 0 and phiy == 1:
    angle = "φ"
  if phix == 1 and phiy == 2:
    angle = "θ"
  if phix == 2 and phiy == 0:
    angle = "Ψ"
  plt.ylabel(angle, fontdict={'fontsize': 4*scale}, rotation='horizontal')

  if save_to_file:
    plt.savefig(filename + ".png", bbox_inches='tight', pad_inches=0.2)
    plt.close()
  else:
    plt.show()


# def print_event_3d_2phi(event, filename='event_3d_2phi'):
#   fig = plt.figure(figsize=(16*plotscale, 9*plotscale))
#   ax = plt.axes(projection='3d')
#
#   ax.scatter3D(
#     [h[x] for h in event.hits],
#     [h[y] for h in event.hits],
#     [hit_phi(h, h.x, h.y) for h in event.hits],
#     color=default_color,
#     s=2*scale
#   )

def print_event_3d_phi(event, tracks, x=2, y=0, phix=0, phiy=1, track_color=0, rotate=False, filename='event_3d_phi', save_to_file=False):
  fig = plt.figure(figsize=(16*plotscale, 9*plotscale))
  ax = plt.axes(projection='3d')

  if len(tracks) > 0:
    count = 0
    for t in [t for t in tracks if len(t.hits)>=3]:
      ax.scatter3D(
        [h[x] for h in t.hits],
        [h[y] for h in t.hits],
        [hit_phi(h[phix], h[phiy]) for h in t.hits],
        color=colors[count % len(colors)],
        s=2*scale
      )

      ax.plot3D(
        [h[x] for h in t.hits],
        [h[y] for h in t.hits],
        [hit_phi(h[phix], h[phiy]) for h in t.hits],
        color=colors[count % len(colors)],
        linewidth=1
      )
      count += 1

  plt.tick_params(axis='both', which='major', labelsize=4*scale)
  ax.set_xlabel(f'{ntox[x]}', fontdict={'fontsize': 4*scale})
  ax.set_ylabel(f'{ntox[y]}', fontdict={'fontsize': 4*scale})
  angle = ""
  if phix == 0 and phiy == 1:
    angle = "φ"
  if phix == 1 and phiy == 2:
    angle = "θ"
  if phix == 2 and phiy == 0:
    angle = "Ψ"
  ax.set_zlabel(angle, fontdict={'fontsize': 4*scale}, rotation='horizontal')

  if save_to_file:
    plt.savefig(filename + ".png", bbox_inches='tight', pad_inches=0.2)
    plt.close()
  else:
    if rotate:
      for angle in range(0, 360):
        ax.view_init(10, angle)
        plt.draw()
        plt.pause(.001)
    else:
      plt.show()


def print_event_3d_3phi(event, tracks , track_color=0, rotate=False, filename='event_3d_3phi', save_to_file=False):
  fig = plt.figure(figsize=(16*plotscale, 9*plotscale))
  ax = plt.axes(projection='3d')

  if len(tracks) > 0:
    count = 0
    for t in [t for t in tracks if len(t.hits)>=3]:
      ax.scatter3D(
        [hit_phi(h[0], h[1]) for h in t.hits],
        [hit_phi(h[1], h[2]) for h in t.hits],
        [hit_phi(h[2], h[0]) for h in t.hits],
        color=colors[count % len(colors)],
        s=2*scale
      )

      ax.plot3D(
        [hit_phi(h[0], h[1]) for h in t.hits],
        [hit_phi(h[1], h[2]) for h in t.hits],
        [hit_phi(h[2], h[0]) for h in t.hits],
        color=colors[count % len(colors)],
        linewidth=1
      )
      count += 1

  plt.tick_params(axis='both', which='major', labelsize=4*scale)
  ax.set_xlabel("φ", fontdict={'fontsize': 4*scale})
  ax.set_ylabel("θ", fontdict={'fontsize': 4*scale})
  ax.set_zlabel("Ψ", fontdict={'fontsize': 4*scale}, rotation='horizontal')

  if save_to_file:
    plt.savefig(filename + ".png", bbox_inches='tight', pad_inches=0.2)
    plt.close()
  else:
    if rotate:
      for angle in range(0, 360):
        ax.view_init(10, angle)
        plt.draw()
        plt.pause(.001)
    else:
      plt.show()
