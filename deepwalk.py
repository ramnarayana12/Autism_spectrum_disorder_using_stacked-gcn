import numpy as np
import networkx as nx
from gensim.models import Word2Vec

class deepWalk:
    def __init__(self, graph, walk_length=10, num_walks=80, embedding_dim=64, window_size=5, workers=4):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.workers = workers

    def generate_random_walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                current_node = node
                for _ in range(self.walk_length - 1):
                    neighbors = list(self.graph.neighbors(current_node))
                    if len(neighbors) == 0:
                        break
                    current_node = np.random.choice(neighbors)
                    walk.append(current_node)
                walks.append(walk)
        return walks

    def get_embeddings(self):
        walks = self.generate_random_walks()
        walks = [list(map(str, walk)) for walk in walks]  # Convert nodes to strings
        model = Word2Vec(
            walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,  # Use Skip-gram
            workers=self.workers
        )
        embeddings = np.array([model.wv[str(node)] for node in self.graph.nodes()])
        return embeddings