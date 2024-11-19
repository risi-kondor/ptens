import ptens
import pytest
import torch
from conftest import get_graph_list
from ptens.exception import GraphCreationError

def get_graph_cache_id_list():
    return [(i, g) for i, g in enumerate(get_graph_list())]
        
@pytest.mark.parametrize("idx, graph", get_graph_cache_id_list())
def test_graph_create_cache(idx, graph):
    graph.cache(idx)
    assert ptens.ggraph.from_cache(idx) == graph


def get_invalid_graph_factory_list():
    class GraphFactory:
        def __init__(self, fn, arg):
            self.fn = fn
            self.arg =arg
        def __call__(self):
            return self.fn(self.arg)
        
        
    return [GraphFactory(ptens.ggraph.from_edge_index, torch.Tensor([[0], [0]]).int()), #Simplest graph
            GraphFactory(ptens.ggraph.from_edge_index, torch.Tensor( # star
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 2, 3, 4, 5, 0, 7, 8, 9, 6,]]).int()),
            GraphFactory(ptens.ggraph.from_matrix, torch.Tensor([[1, 1, 0], [1, 0, 1], [0, 1, 0]]).int()), # From Matrix
            ]

@pytest.mark.parametrize("graph_factory", get_invalid_graph_factory_list())
def test_invalid_graphs(graph_factory):
    with pytest.raises(GraphCreationError):
        graph_factory()

