import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_curve, auc, f1_score
from rdkit.Chem import MACCSkeys

import networkx as nx
from torch_geometric.utils import from_networkx
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import numpy as np

from rdkit import Chem


def get_functional_group_features(smiles):
    functional_groups = {
        'carboxylic_acid': '[CX3](=O)[OX2H1]',
        'amine': '[NX3;H2,H1;!$(NC=O)]',
        'amide': '[NX3][CX3](=O)',
        'alcohol': '[OX2H]',
        'ketone': '[#6][CX3](=O)[#6]',
        'alkene': '[CX3]=[CX3]',
        'nitrile': '[CX1]#[NX2]',
    }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(functional_groups))
    features = []
    for pattern in functional_groups.values():
        features.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))))
    return np.array(features)


def scaffold_split(df, smiles_col='SMILES', target_col='Activity', split_ratios=(0.8, 0.1, 0.1), seed=42):
    scaffolds_to_indices = defaultdict(list)
    for idx, smiles in enumerate(df[smiles_col]):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        scaffolds_to_indices[scaffold].append(idx)

    positive_indices = df[df[target_col] == 1].index.tolist()
    negative_indices = df[df[target_col] == 0].index.tolist()

    np.random.seed(seed)
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)

    pos_train_end = int(len(positive_indices) * split_ratios[0])
    pos_valid_end = pos_train_end + int(len(positive_indices) * split_ratios[1])

    neg_train_end = int(len(negative_indices) * split_ratios[0])
    neg_valid_end = neg_train_end + int(len(negative_indices) * split_ratios[1])

    train_indices = positive_indices[:pos_train_end] + negative_indices[:neg_train_end]
    valid_indices = positive_indices[pos_train_end:pos_valid_end] + negative_indices[neg_train_end:neg_valid_end]
    test_indices = positive_indices[pos_valid_end:] + negative_indices[neg_valid_end:]

    np.random.shuffle(train_indices)
    np.random.shuffle(valid_indices)
    np.random.shuffle(test_indices)

    return df.iloc[train_indices], df.iloc[valid_indices], df.iloc[test_indices]


class MFAGCN(nn.Module):
    def __init__(self, num_features, num_fingerprint_features, num_func_group_features, num_classes):
        super(MFAGCN, self).__init__()
        # 定义三个GCN层
        self.gcn1 = GCNConv(num_features, 128)
        self.gcn2 = GCNConv(128, 256)
        self.gcn3 = GCNConv(256, 256)
        self.fingerprint_embedding = nn.Linear(num_fingerprint_features, 256)
        self.func_group_embedding = nn.Linear(num_func_group_features, 256)
        self.attention_weights = nn.Parameter(torch.rand(256, 256))
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.residual1 = nn.Linear(128, 256)
        self.residual2 = nn.Linear(256, 256)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        fingerprint_features = data.fingerprint_features.squeeze(1)
        func_group_features = data.func_groups.squeeze(1)

        x1 = F.relu(self.gcn1(x, edge_index))

        x2 = F.dropout(x1, p=0.5, training=self.training)
        x2 = F.relu(self.gcn2(x2, edge_index) + self.residual1(x1))
        x3 = F.relu(self.gcn3(x2, edge_index) + self.residual2(x2))

        fingerprint_embedded = F.relu(self.fingerprint_embedding(fingerprint_features))
        func_group_embedded = F.relu(self.func_group_embedding(func_group_features))

        attention_scores = torch.matmul(x3, self.attention_weights)
        attention_scores = F.softmax(attention_scores, dim=1)
        x = attention_scores * x3
        x = global_mean_pool(x, batch)

        x += fingerprint_embedded + func_group_embedded
        x = self.fc(x)

        return x.view(-1, 1)


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    g = nx.Graph()

    node_features = []
    for atom in mol.GetAtoms():
        feature = torch.tensor([atom.GetAtomicNum()], dtype=torch.float)
        g.add_node(atom.GetIdx())
        node_features.append(feature.unsqueeze(0))

    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    data = from_networkx(g)
    data.x = torch.cat(node_features, dim=0)
    return data


def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(1, 166)
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    fp_array = np.array(fingerprint)
    return torch.tensor(fp_array[1:], dtype=torch.float32).unsqueeze(0)


def prepare_dataset(dataframe):
    dataset = []
    for index, row in dataframe.iterrows():
        graph = smiles_to_graph(row['SMILES'])
        if graph:
            fingerprint = smiles_to_maccs(row['SMILES'])
            func_groups = torch.tensor(get_functional_group_features(row['SMILES']), dtype=torch.float32).unsqueeze(0)
            if fingerprint is not None:
                graph.fingerprint_features = fingerprint
                graph.func_groups = func_groups
                graph.y = torch.tensor([row['Activity']], dtype=torch.float)
                dataset.append(graph)
    print("Dataset size:", len(dataset))
    return dataset


df = pd.read_csv('abucin_classify.csv')

train_df, valid_df, test_df = scaffold_split(df)
train_dataset = prepare_dataset(train_df)
valid_dataset = prepare_dataset(valid_df)
test_dataset = prepare_dataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MFAGCN(num_features=1, num_fingerprint_features=166, num_func_group_features=7, num_classes=1).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)


def calculate_class_weights(train_df, target_col):
    class_counts = train_df[target_col].value_counts()
    weights = {cls: (sum(class_counts) / count) ** 0.5 for cls, count in class_counts.items()}
    weights_tensor = torch.tensor([weights[i] for i in sorted(weights)], dtype=torch.float)
    return weights_tensor


class_weights = calculate_class_weights(train_df, 'Activity')
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1] / class_weights[0]).to(device)


def train():
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.unsqueeze(1))  # 使用加权损失函数
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(loader):
    model.eval()
    y_true, y_scores, y_pred = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            y_true.extend(data.y.tolist())
            y_scores.extend(output.flatten().tolist())
            y_pred.extend((output.flatten() > 0).int().tolist())
    print("Actual y_true in batch:", y_true)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    if len(set(y_true)) == 1:
        print("Warning: Only one class present in y_true.")
    auprc = auc(recall, precision)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    return auprc, f1


for epoch in range(500):
    loss = train()
    auprc, f1 = evaluate(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {loss}, AUPRC: {auprc}, F1 Score: {f1}')

test_auprc, test_f1 = evaluate(test_loader)
print(f'Test AUPRC: {test_auprc}, Test F1 Score: {test_f1}')
