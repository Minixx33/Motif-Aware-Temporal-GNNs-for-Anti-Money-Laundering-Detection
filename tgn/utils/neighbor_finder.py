import numpy as np

class NeighborFinder:
    """
    Stores adjacency lists so TGN can retrieve temporal neighbors.
    This matches the original Jodie/TGN behavior.
    """

    def __init__(self, adj_list, uniform=True):
        """
        adj_list: list of lists, each entry:
                  [(neighbor, edge_id, timestamp), ...]
        """
        self.adj_list = adj_list
        self.uniform = uniform

    def find_before(self, neighbors, cut_time):
        """
        Returns all neighbors with timestamp < cut_time.
        """
        return [n for n in neighbors if n[2] < cut_time]

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Retrieves temporal neighbors for each node.
        Returns:
            neighbors (N x n_neighbors)
            edge_idxs (N x n_neighbors)
            times     (N x n_neighbors)
        """
        out_neighbors = []
        out_edge_idxs = []
        out_timestamps = []

        for src, ts in zip(source_nodes, timestamps):
            neighbors = self.adj_list[src]
            past_neighbors = self.find_before(neighbors, ts)

            if len(past_neighbors) == 0:
                # No neighbors -> return dummy 0
                out_neighbors.append([0] * n_neighbors)
                out_edge_idxs.append([0] * n_neighbors)
                out_timestamps.append([0] * n_neighbors)
                continue

            # Sort by most recent
            past_neighbors = sorted(past_neighbors, key=lambda x: x[2], reverse=True)

            # Take most recent N
            sampled = past_neighbors[:n_neighbors]

            # If fewer, pad with zeros
            while len(sampled) < n_neighbors:
                sampled.append((0, 0, 0))

            neigh_only  = [x[0] for x in sampled]
            eidx_only   = [x[1] for x in sampled]
            time_only   = [x[2] for x in sampled]

            out_neighbors.append(neigh_only)
            out_edge_idxs.append(eidx_only)
            out_timestamps.append(time_only)

        return (
            np.array(out_neighbors),
            np.array(out_edge_idxs),
            np.array(out_timestamps),
        )
