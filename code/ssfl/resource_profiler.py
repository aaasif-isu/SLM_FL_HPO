import numpy as np
import json


def simulate_latency(num_clients, num_clusters):
    """
    Simulate latency values for clients and assign them to clusters.
    Lower latency = higher-end device.
    """
    assert num_clusters <= num_clients, "Number of clusters must be <= number of clients"

    # Define cluster centers (spaced across realistic latency range)
    latency_centers = np.linspace(300, 50, num_clusters)  # From slow (300ms) to fast (50ms)
    cluster_sizes = [num_clients // num_clusters] * num_clusters
    for i in range(num_clients % num_clusters):
        cluster_sizes[i] += 1  # distribute leftovers

    latencies = []
    for c in range(num_clusters):
        latencies.extend(np.random.normal(loc=latency_centers[c], scale=10, size=cluster_sizes[c]))

    latencies = np.array(latencies)
    shuffled_indices = np.random.permutation(num_clients)
    latencies = latencies[shuffled_indices]

    return latencies


def assign_clusters_from_latency(latencies, num_clusters):
    """
    Assign clusters using quantile-based thresholding from latency values.
    """
    thresholds = np.quantile(latencies, np.linspace(0, 1, num_clusters + 1)[1:-1])
    clusters = np.digitize(latencies, thresholds, right=True)  # returns values from 0 to (num_clusters-1)
    return clusters.tolist()


def profile_resources(num_clients, num_clusters, output_path=None):
    latencies = simulate_latency(num_clients, num_clusters)
    clusters = assign_clusters_from_latency(latencies, num_clusters)

    profile = {
        f"client_{i}": {
            "latency": float(latencies[i]),
            "cluster": int(clusters[i])
        }
        for i in range(num_clients)
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=4)

    return clusters