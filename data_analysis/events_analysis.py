import json
import math
import os

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import sys

# this gets the full path of the project root on your pc and adds it to the path
project_root = os.path.abspath(os.path.join(__file__,'..','..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import event_model.event_model as em
from algorithms import clustering
from algorithms.clustering import Clustering
from validator.validator_lite import MCParticle

# ----------------------------------- UTILS ------------------------------------------------
def get_bins(x):
    q25, q75 = np.percentile(x, [.25, .75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((x.max() - x.min()) / bin_width)
    print("Freedman–Diaconis number of bins:", bins)
    return bins


def get_json_data_from_folder(data_set_folder):
    jsons = []
    # for (dirpath, dirnames, filenames) in os.walk(f"../events/{data_set_folder}"):
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(project_root, f"events/{data_set_folder}")):
        for i, filename in enumerate(filenames):
            # Get an event
            # print(f'opening: {filename}')
            f = open(os.path.realpath(os.path.join(dirpath, filename)))
            json_data = json.loads(f.read())
            event = em.event(json_data)
            f.close()
            # print(f'closing : {filename}')

            jsons.append(json_data)
    return jsons


def tracks_from_data(json_data, only_reconstructible=True):
    reconstructible_tracks = []
    hits = []
    for hid, (x, y, z) in enumerate(zip(json_data["x"], json_data["y"], json_data["z"])):
        hits.append(em.hit(x, y, z, hid))
    description = json_data["montecarlo"]["description"]
    particles = json_data["montecarlo"]["particles"]
    for p in particles:
        d = {description[i]: p[i] for i in range(len(description))}
        track_hits = [hits[hit_number] for hit_number in d["hits"]]
        mcp = MCParticle(d.get("key", 0), d.get("pid", 0), d.get("p", 0), d.get("pt", 0), d.get("eta", 0),
                         d.get("phi", 0), d.get("charge", 0), track_hits)

        if only_reconstructible:
            if len(track_hits) >= 3:
                reconstructible_tracks.append(em.track(track_hits))
        else:
            reconstructible_tracks.append(em.track(track_hits))
    return reconstructible_tracks


def noise_from_data(json_data):
    noise = 0
    hits_in_reconstructible_tracks = 0
    tracks = []
    hits = []
    for hid, (x, y, z) in enumerate(zip(json_data["x"], json_data["y"], json_data["z"])):
        hits.append(em.hit(x, y, z, hid))
    description = json_data["montecarlo"]["description"]
    particles = json_data["montecarlo"]["particles"]
    for p in particles:
        d = {description[i]: p[i] for i in range(len(description))}
        track_hits = [hits[hit_number] for hit_number in d["hits"]]
        if len(track_hits) >= 3:
            hits_in_reconstructible_tracks += len(track_hits)

    noise = len(hits) - hits_in_reconstructible_tracks
    return noise


def origins_from_data(json_data, limit=False):
    origins = []
    hits = []
    for hid, (x, y, z) in enumerate(zip(json_data["x"], json_data["y"], json_data["z"])):
        hits.append(em.hit(x, y, z, hid))
    description = json_data["montecarlo"]["description"]
    particles = json_data["montecarlo"]["particles"]
    for p in particles:
        d = {description[i]: p[i] for i in range(len(description))}
        track_hits = [hits[hit_number] for hit_number in d["hits"]]
        if limit:
            if -15 <= track_hits[0].x <= 15 and -10 <= track_hits[0].y <= 10:
                origins.append(track_hits[0])
        else:
            origins.append(track_hits[0])

    return origins


# ----------------------------------- UTILS ------------------------------------------------


# ----------------------------------- PLOTS ------------------------------------------------
def data_distribution(data):
    total = 0
    for num in data:
        total += num

    mu = total / len(data)

    aux = 0
    for num in data:
        aux += ((num - mu) ** 2)

    variance = aux / len(data)
    sigma = math.sqrt(variance)

    print(f'Mean: {mu}')
    print(f'Variance: {variance}')
    print(f'Standard deviation: {sigma}')

    return mu, variance, sigma


def plot_distribution(mu, sigma, title='Distribution', safe_to_file=False):
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label='pdf')
    plt.title(title)
    plt.text(-300, .0020, fr'$\mu={round(mu, 4)},\ \sigma={round(sigma, 4)}$')
    # plt.plot(x, stats.norm.cdf(x, mu, sigma), label='cdf')
    plt.legend()
    plt.grid(True)
    if safe_to_file:
        plt.savefig(f'event_plots/{title}', bbox_inches='tight', pad_inches=0.2)
        plt.close()
    else:
        plt.show()


def plot_density_histogram(data, title="Histogram", xlabel="Data", ylabel="", safe_to_file=False):
    x = np.array(data)
    plt.hist(x, density=True, bins=get_bins(x), label="Data")

    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = stats.gaussian_kde(x)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")

    plt.legend(loc="upper left")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if safe_to_file:
        plt.savefig(f'event_plots/{title}', bbox_inches='tight', pad_inches=0.2)
        plt.close()
    else:
        plt.show()


def plot_histogram(data, title="Histogram", xlabel="Data", ylabel="", safe_to_file=False):
    x = np.array(data)
    plt.hist(x, get_bins(x), label="Data")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if safe_to_file:
        plt.savefig(f'event_plots/{title}', bbox_inches='tight', pad_inches=0.2)
        plt.close()
    else:
        plt.show()
# ----------------------------------- PLOTS ------------------------------------------------


# ------------------------------- DATA ANALYSIS --------------------------------------------
def distribution_of_tracks(safe_to_file=False):
    events_tracks = []

    jsons = get_json_data_from_folder(data_set)
    for json_data in jsons:
        events_tracks.append(len(tracks_from_data(json_data=json_data)))

    mu, variance, sigma = data_distribution(events_tracks)
    plot_distribution(mu, sigma, f'Distribution of #tracks in {data_set}', safe_to_file=safe_to_file)


def distribution_of_noise(safe_to_file=False):
    events_noise = []

    jsons = get_json_data_from_folder(data_set)
    for json_data in jsons:
        events_noise.append(noise_from_data(json_data=json_data))

    mu, variance, sigma = data_distribution(events_noise)
    plot_distribution(mu, sigma, f'Distribution of #noise in {data_set}', safe_to_file=safe_to_file)


def noise_histogram(density=False, safe_to_file=False):
    events_noise = []

    jsons = get_json_data_from_folder(data_set)
    for json_data in jsons:
        events_noise.append(noise_from_data(json_data=json_data))

    if density:
        plot_density_histogram(events_noise, title=f'#noise density histogram in {data_set}', xlabel="#noise",
                               ylabel="Probability", safe_to_file=safe_to_file)
    else:
        plot_histogram(events_noise, title=f'#noise histogram in {data_set}', xlabel="#noise", safe_to_file=safe_to_file)


def tracks_histogram(density=False, safe_to_file=False):
    events_tracks = []
    jsons = get_json_data_from_folder(data_set)
    for json_data in jsons:
        events_tracks.append(len(tracks_from_data(json_data=json_data)))

    if density:
        plot_density_histogram(events_tracks, title=f'#tracks density histogram in {data_set}', xlabel="#tracks",
                               ylabel="Probability", safe_to_file=safe_to_file)
    else:
        plot_histogram(events_tracks, title=f'#tracks histogram in {data_set}', xlabel="#tracks", safe_to_file=safe_to_file)


def tracks_by_noise(safe_to_file=False):
    events_noise = []
    events_tracks = []

    jsons = get_json_data_from_folder(data_set)
    for json_data in jsons:
        events_noise.append(noise_from_data(json_data=json_data))
        events_tracks.append(len(tracks_from_data(json_data=json_data)))

    x = np.array(events_tracks)
    y = np.array(events_noise)

    plt.scatter(x, y, s=1, marker='o')

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, c='y', linewidth=0.5)

    plt.xlabel('#tracks')
    plt.ylabel('#noise')
    title = f'#tracks by #noise on {data_set}'
    plt.title(title)
    plt.grid(True)
    if safe_to_file:
        plt.savefig(f'event_plots/{title}', bbox_inches='tight', pad_inches=0.2)
        plt.close()
    else:
        plt.show()


def plot_divided_tracks(safe_to_file=False):
    x = np.arange(start=1, stop=17, step=1)
    y1 = np.array([0, 5.4093, 9.0091, 9.621, 12.6665, 12.4799, 15.4293, 15.3473, 18.0204, 17.9006, 20.1307, 20.2586, 22.2226, 22.3971, 24.4066, 24.2933])
    y2 = np.array([0, 5.2338, 8.7725, 9.3465, 12.42, 12.1994, 15.1068, 11.3658, 14.4671, 14.3217, 16.938, 16.7929, 19.1647, 19.0381, 21.23, 21.0619])
    plt.scatter(x, y1, marker='s', s=5, c='teal', label='minibias')
    plt.scatter(x, y2, marker='o', s=5, c='darkred', label='bsphiphi')

    m1, b1 = np.polyfit(x, y1, 1)
    m2, b2 = np.polyfit(x, y2, 1)
    plt.plot(x, m1*x + b1, c='skyblue', alpha=0.5, linestyle='dashed')
    plt.plot(x, m2*x + b2, c='salmon', alpha=0.5, linestyle='dashed')

    plt.legend(loc='upper left')
    plt.xlabel('#clusters')
    plt.ylabel('%divided tracks')
    title = 'tracks divided by clusters'
    plt.title(title)
    if safe_to_file:
        plt.savefig(f'event_plots/{title}', bbox_inches='tight', pad_inches=0.2)
        plt.close()
    else:
        plt.show()


def divided_tracks():
    jsons = get_json_data_from_folder(data_set)

    for k in range(2, 17):
        total_tracks = 0
        total_divided_tracks = 0
        for i in range(0, len(jsons)):
            json_data = jsons[i]
            event = em.event(json_data)
            bins = Clustering(K=k).get_bins(event)
            tracks = tracks_from_data(json_data)

            out_of_bin = 0

            for track in tracks:
                hits_in_bins = [0 for b in range(len(bins))]
                for hit in track.hits:
                    for i, b in enumerate(bins):
                        if hit in b:
                            hits_in_bins[i] += 1

                for b in hits_in_bins:
                    if b != 0 and b != sum(hits_in_bins):
                        out_of_bin += 1

            total_tracks += len(tracks)
            total_divided_tracks += out_of_bin

        print()
        print(f'with {k} bins\n'
              f'divided tracks = {total_divided_tracks}\n'
              f'form = {total_tracks}\n'
              f'{round(total_divided_tracks * 100 / total_tracks, 4)}%')


def track_origin_analysis():
    events_origins = []
    events_origins_limited = []

    jsons = get_json_data_from_folder(data_set)
    for json_data in jsons:
        events_origins.append(origins_from_data(json_data=json_data))
        events_origins_limited.append(origins_from_data(json_data=json_data, limit=True))

    total_origins = 0
    events_origin_means = []
    percentages = []
    for e in range(len(events_origins)):
        percentages.append(len(events_origins_limited[e])/len(events_origins[e]))

        event_sum = np.zeros(2)
        for origin in events_origins[e]:
            event_sum = event_sum + np.array([origin[0], origin[1]])
        event_mean = event_sum / len(events_origins[e])
        events_origin_means.append(event_mean)

    for p in percentages:
        print(p)
# ------------------------------- DATA ANALYSIS --------------------------------------------


if __name__ == '__main__':
    data_set = "minibias"

    distribution_of_noise(safe_to_file=True)
    noise_histogram(safe_to_file=True)
    noise_histogram(density=True, safe_to_file=True)

    distribution_of_tracks(safe_to_file=True)
    tracks_histogram(safe_to_file=True)
    tracks_histogram(density=True, safe_to_file=True)

    tracks_by_noise(safe_to_file=True)

    data_set = "bsphiphi"

    distribution_of_noise(safe_to_file=True)
    noise_histogram(safe_to_file=True)
    noise_histogram(density=True, safe_to_file=True)

    distribution_of_tracks(safe_to_file=True)
    tracks_histogram(safe_to_file=True)
    tracks_histogram(density=True, safe_to_file=True)

    tracks_by_noise(safe_to_file=True)
