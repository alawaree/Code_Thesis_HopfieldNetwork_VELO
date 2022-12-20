# Marius analytics script
# things I want to analyse / plot

# fwd / bakward / noise hits per module -> I know we have 52
# distr first hits
# plot of single events points in polar (unroll)
# max polar change p track (over all tracks) (between 2 consec modules (does this mean 1 or 2 now?!))
# per track -> smallest r particle -> from there we can also see in which direction the particle is flying

# get the relevant dependencies 
import json
import os
import sys 
import random
import math

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import quantiles

# this gets the full path of the project root on your pc and adds it to the path
project_root = os.path.abspath(os.path.join(__file__,'..','..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import event_model.event_model as em

# resource switch
dataset_folder={
    'small':'events/small_dataset',
    'bsphiphi': 'events/bsphiphi',
    'minibias': 'events/minibias'
}

# inherit classes and alter
def get_events_data_from_folder(data_set_folder, num_events = -1, shuffle = False):
    events = []
    # for (dirpath, dirnames, filenames) in os.walk(f"../events/{data_set_folder}"):
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(project_root, data_set_folder)):
        if shuffle:
            random.shuffle(filenames)
        if num_events > 0 or num_events < len(filenames):
            filenames = filenames[:num_events]
        for i, filename in enumerate(filenames):
            # Get an event
            print(f'opening: {filename}')
            f = open(os.path.realpath(os.path.join(dirpath, filename)))
            json_data = json.loads(f.read())
            event = em.event(json_data,read_tracks=True)
            events.append(event)
            f.close()
            print(f'closing : {filename}')
    return events

def update_event(event):
    for hit in event.hits:
        hit.update_polar()
    
    for track in event.real_tracks:
        track.update_track()
    

def update_all_events(events):
    for event in events:
        update_event(event)

def print_multi_event_hit_directions(events, dataset):
    fh_counter = np.zeros(shape=(2,52))
    counters = np.zeros(shape=(3,52))
    for event in events:
        for hit in event.hits:
            counters[hit.direction + 1, hit.module_number] += 1

        for track in event.real_tracks:
            direction = 0
            if track.direction == -1:
                direction = 1
            fh_counter[direction, track.first_hit_module] += 1

    fig = plt.figure(figsize=(8,8))
    ax1= fig.add_subplot(211)
    ax2= fig.add_subplot(212)

    x = np.array(range(52))
    ax1.bar(x,counters[0,:],width=0.5, label='Left')
    ax1.bar(x,counters[2,:],width=0.5,bottom = counters[0,:], label='Reft')
    ax1.bar(x,counters[1,:],width=0.5,bottom = counters[0,:]+ counters[2,:], label='Noise')
    ax1.legend()
    ax1.set_title('Track hit distribution couloured by trajectory direction')

    ax2.bar(x, fh_counter[0,:], width=0.5, label='To Right')
    ax2.bar(x, fh_counter[1,:],bottom = fh_counter[0,:], width=0.5, label='To Left')
    ax2.legend()
    ax2.set_title('Tracks first hit modules coloured by trajectory direction')
    #plt.show()
    plt.savefig(os.path.join(project_root, f'data_analysis/polar_analysis_plots/hit_directions_{dataset}.png'))

def print_multi_event_track_phi_deviations(events, dataset):
    largest_phi_changes = []
    largest_phi_cons_changes = []

    min_r = []
    for event in events:
        for track in event.real_tracks:
            largest_phi_changes.append(track.max_phi_change)
            largest_phi_cons_changes.append(track.max_cosecutive_phi_change)
            min_r.append(track.min_r)
     

    # sort the data:
    data = np.array(largest_phi_changes)
    data_consecutive = np.array(largest_phi_changes)
    data_sorted = np.sort(largest_phi_changes )
    
    print(np.array(quantiles(min_r ,n=10)))
    pd.DataFrame([quantiles(min_r ,n=10),quantiles(largest_phi_changes ,n=10),quantiles(largest_phi_cons_changes ,n=10)], columns= [f'{10*i}th' for i in range(1,10)], 
        index=['min_r','max_phi_dev', 'max_phi_in_consecutive']).to_csv('data_analysis/polar_analysis_plots/quantiles.csv')


if __name__ == '__main__':
    dataset = 'bsphiphi'
    events = get_events_data_from_folder(dataset_folder[dataset], num_events = 10, shuffle=False)
    update_all_events(events)
    print_multi_event_hit_directions(events,dataset)
    print_multi_event_track_phi_deviations(events,dataset)

