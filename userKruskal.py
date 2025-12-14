import networkx as nx
import matplotlib.pyplot as plt
import random
import time


# 1. Disjoint Set (Union-Find) for Kruskal


class DisjointSet:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, v):
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, u, v):
        root1 = self.find(u)
        root2 = self.find(v)

        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
            return True
        return False


# 2. Generate Random Graph (Sparse or Dense)


def generate_graph(num_nodes, edge_prob, min_weight=1, max_weight=10):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                G.add_edge(i, j, weight=random.randint(min_weight, max_weight))
    return G


# 3. Custom Kruskal MST


def kruskal_mst(num_nodes, edges, visualize_steps=False, pos=None, G=None):
    ds = DisjointSet(range(num_nodes))
    mst = []
    total_weight = 0

    sorted_edges = sorted(edges, key=lambda x: x[2])  # (u, v, w)

    if visualize_steps:
        plt.figure(figsize=(7, 7))

    step = 1

    for u, v, w in sorted_edges:
        if ds.union(u, v):
            mst.append((u, v, w))
            total_weight += w

            if visualize_steps:
                plt.clf()

                # Draw original graph with ALL edges + labels
                nx.draw(G, pos,
                        with_labels=True,
                        edge_color='lightgray',
                        node_color='lightblue',
                        node_size=600)

                # Draw edge weight labels for the whole graph
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

                # Draw MST edges in red
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(x[0], x[1]) for x in mst],
                    edge_color='red',
                    width=2
                )

                plt.title(f"MST Step {step}: Added ({u}, {v}) weight={w}")
                plt.pause(1)
                step += 1

        if len(mst) == num_nodes - 1:
            break

    if visualize_steps:
        plt.show()

    return mst, total_weight



# 4. Analyze Graph (execution time, edges, mst stats)


def analyze_graph(G, label):
    edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    num_nodes = G.number_of_nodes()

    start = time.time()
    mst, weight = kruskal_mst(num_nodes, edges)
    end = time.time()

    return {
        "type": label,
        "nodes": num_nodes,
        "edge_count": len(edges),
        "mst_edge_count": len(mst),
        "mst_total_weight": weight,
        "execution_time": round(end - start, 6)
    }

# 5. Side-by-Side Graph Visualization
# 

def display_sparse_and_dense(sparse_G, dense_G):
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    nx.draw(sparse_G, with_labels=True, node_color="lightgreen")
    plt.title("Sparse Graph")

    plt.subplot(122)
    nx.draw(dense_G, with_labels=True, node_color="lightcoral")
    plt.title("Dense Graph")

    plt.show()

#
# 6. MAIN PROGRAM
# 

if __name__ == "__main__":
    NUM_NODES = 10

    # Generate graphs
    sparse = generate_graph(NUM_NODES, 0.2)
    dense = generate_graph(NUM_NODES, 0.7)

    # Display both graphs side-by-side
    display_sparse_and_dense(sparse, dense)

    # Analyze performance for both
    sparse_results = analyze_graph(sparse, "Sparse")
    dense_results = analyze_graph(dense, "Dense")

    print("\n=== PERFORMANCE RESULTS ===")
    print(sparse_results)
    print(dense_results)

    # 
    # Step-by-step MST for BOTH GRAPHS
    # 

    print("\nVisualizing MST Formation (Sparse Graph)")
    sparse_pos = nx.spring_layout(sparse, seed=42)
    sparse_edges = [(u, v, d['weight']) for u, v, d in sparse.edges(data=True)]
    kruskal_mst(NUM_NODES, sparse_edges, visualize_steps=True, pos=sparse_pos, G=sparse)

    print("\nVisualizing MST Formation (Dense Graph)")
    dense_pos = nx.spring_layout(dense, seed=42)
    dense_edges = [(u, v, d['weight']) for u, v, d in dense.edges(data=True)]
    kruskal_mst(NUM_NODES, dense_edges, visualize_steps=True, pos=dense_pos, G=dense)
