from src.utils import *
from src.model import *
from functools import partial

# --------------------------------- ARGPARSE --------------------------------- #

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, help="type of GNN layer")
parser.add_argument('--full', dest="subset", action="store_false", help="run full ZINC")
parser.add_argument('--subset', dest="subset", action="store_true", help="only subset")
parser.add_argument("--test", action="store_true", dest="test")

parser.add_argument("--seed", type=int, default=19260817, help="random seed")
parser.add_argument("--indir", type=str, default="data/zinc", help="dataset")
parser.add_argument("--outdir", type=str, default="result.zinc", help="output")
parser.add_argument("--device", type=int, default=None, help="CUDA device")

parser.add_argument("--max_dis", type=int, default=5, help="distance encoding")
parser.add_argument("--num_layer", type=int, default=6, help="number of layers")
parser.add_argument("--dim_embed", type=int, default=96, help="embedding dimension")

parser.add_argument("--bs", type=int, default=128, help="batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epochs", type=int, default=400, help="training epochs")

args = parser.parse_args()
print(f"""Run:
    model: {args.model}
    subset: {args.subset}
    seed: {args.seed}
""")

if args.model == "MP":
    args.dim_embed *= 11
    args.dim_embed //= 7
    args.max_dis = 0
elif args.model == "Sub-G":
    args.dim_embed *= 5
    args.dim_embed //= 4
        

id = f"{args.model}-{args.subset}-{args.max_dis}-{args.num_layer}x{args.dim_embed}-{args.bs}-{args.lr}-{args.seed}"

torch.manual_seed(args.seed)
if args.device is None: device = torch.device("cpu") 
else: device = torch.device(f"cuda:{args.device}") 

# ---------------------------------- DATASET --------------------------------- #

from src import dataset

import torch
from itertools import chain

def select_cycles(cycles):
    selected_cycles = []
    vertex_to_cycles = {}

    def can_select_cycle(cycle):
        shared_vertices = 0
        for vertex in cycle:
            if vertex in vertex_to_cycles and vertex_to_cycles[vertex] > 0:
                shared_vertices += 1
            if shared_vertices > 1:
                return False
        return True

    def update_vertex_count(cycle, increment):
        for vertex in cycle:
            if vertex in vertex_to_cycles:
                vertex_to_cycles[vertex] += increment
            else:
                vertex_to_cycles[vertex] = increment

    for cycle in cycles:
        if can_select_cycle(cycle):
            selected_cycles.append(cycle)
            update_vertex_count(cycle, 1)

    return selected_cycles

def count_connected_components(graph, num_nodes):
    def dfs(node, visited):
        stack = [node]
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

    visited = set()
    connected_components = 0
    for node in range(num_nodes):
        if node not in visited:
            dfs(node, visited)
            connected_components += 1

    return connected_components

def find_n_length_cycles(graph, n):

    def dfs(current, start, path):
        if len(path) == n:
            if start in graph[current]:
                cycles.append(path[:])
            return

        for neighbor in graph[current]:
            if neighbor not in path:
                path.append(neighbor)
                dfs(neighbor, start, path)
                path.pop()

    cycles = []
    for node in graph:
        dfs(node, node, [node])
        # graph[node] = []

    for cycle in cycles:
      cycle.sort()
    cycleset=set(tuple(i) for i in cycles)
    return cycleset


# Define a function to add different feature values to each node
def add_node_features(data, feature_values):
    assert len(feature_values) == data.num_nodes, "Feature values tensor length must match the number of nodes"
    new_feature = feature_values.clone().detach().view(-1, 1)  # Ensure new_feature is a column vector
    data.x = torch.cat([data.x, new_feature], dim=1)  # Concatenate the new feature to the existing features
    # print(data.x)
    return data


modified_dataset = []
def create_cycle_bases_feature(data,i):
    n = data.num_nodes
    m = len(data.edge_index[0])
    edges = []
    for it in range(m):
      edges.append([data.edge_index[0][it].item(),data.edge_index[1][it].item()])

    new_features=[0]*n
    graph = {}
    for u, v in edges:
      if u not in graph:
        graph[u] = []
      if v not in graph:
        graph[v] = []
      graph[u].append(v)
      graph[v].append(u)

    cycle=find_n_length_cycles(graph, i)
    newcycle=select_cycles(cycle)
    values=set(chain.from_iterable(cycle))
    for it in values:
      new_features[it]=i
    return new_features

dataset_ = ZINC(root='.', subset=True)
for dat in dataset_:
    num_nodes = dat.num_nodes
    for i in range(3,10):
      feature_values = torch.tensor(create_cycle_bases_feature(dat,i))  # Generate cycle bases feature values
      modified_data = add_node_features(dat, feature_values)  # Modify the data object
    modified_dataset.append(modified_data)  # Store the modified data object


dataloader = {
    name: data.dataloader.DataLoader(
        modified_dataset(args.indir,
                     args.subset, name,
                     transform=subgraph),
        batch_size=args.bs,
        num_workers=4,
        shuffle=True
    )
    for name in ["train", "val", "test"]
}



# dataloader = {
#     name: data.dataloader.DataLoader(
#         dataset.ZINC(args.indir,
#                      args.subset, name,
#                      transform=subgraph),
#         batch_size=args.bs,
#         num_workers=4,
#         shuffle=True
#     )
#     for name in ["train", "val", "test"]
# }

# ----------------------------------- MODEL ---------------------------------- #
# copied from OGB mol_encoder.py

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, full_atom_feature_dims):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])
        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim, full_bond_feature_dims=None):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])
        return bond_embedding   
    
    
model = GNN(args, "g",
            enc_a=partial(AtomEncoder, full_atom_feature_dims=[32]),
            enc_e=partial(BondEncoder, full_bond_feature_dims=[5]),
            odim=1)

# ----------------------------------- ITER ----------------------------------- #

def critn(pred, batch):
    pred = pred.squeeze(-1)
    assert pred.shape == (y:=batch.y).shape
    return torch.nn.L1Loss()(pred, y)

def train(model, loader, critn, optim):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        optim.zero_grad()
        loss = critn(pred, batch)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return np.array(losses).mean()

def eval(model, loader, critn):
    model.eval()
    errors = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
          pred = model(batch)
          err = critn(pred, batch)
          errors.append(err.item())
    return np.array(errors).mean()

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                   mode='min',
                                                   factor=0.5,
                                                   patience=20,
                                                #    min_lr=1e-5,
                                                   verbose=True
                                                   )

if args.test:
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name, p.numel())
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    import code;
    exit(code.interact(local=dict(globals(), **dict(locals()))))

# ------------------------------------ RUN ----------------------------------- #

record = []

os.makedirs(output_dir:=args.outdir, exist_ok=True)
assert not os.path.exists(log:=f"{output_dir}/{id}.txt")

from tqdm import trange
for epoch in (pbar:=trange(args.epochs)):
    lr = optim.param_groups[0]['lr']
    loss = train(model, dataloader["train"], critn, optim)
    val = eval(model, dataloader["val"], critn)
    test = eval(model, dataloader["test"], critn)

    sched.step(val)

    record.append([lr, loss, val, test])
    pbar.set_postfix(lr=lr, loss=loss, val=val, test=test)

    np.savetxt(log, record, delimiter='\t')
