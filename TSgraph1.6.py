import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm
import torch
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from scipy.stats import wasserstein_distance
import pickle
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, from_networkx
from tslearn.metrics import dtw
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from tslearn.metrics import dtw
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import networkx as nx


# Import data
df_time = pd.read_csv("")
df_tab1 = pd.read_csv("")
df_tab2 = pd.read_csv("")
df_tab3 = pd.read_csv("")

# Importing LLM
tokenizer, model = load_LMM()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

time_features = df_time.columns.tolist()

# List of tabular datasets
tab_files = []
df_tabs = {}
tab_features_list = {}

for file in tab_files:
    df = pd.read_csv(file)
    df_tabs[file] = df
    tab_features_list[file] = df.columns.tolist()

def get_feature_embeddings(feature_names: list):
    encoded_input = tokenizer(feature_names, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    last_hidden_states = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    embeddings = (last_hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)
    return embeddings

# Obtain the embedding vector of time series features
time_embeddings = get_feature_embeddings(time_features)

# Obtain the embedding vectors of the features of the table to be compared
tab_embeddings = {}
for file, features in tab_features_list.items():
    embeddings = get_feature_embeddings(features)
    tab_embeddings[file] = embeddings

# Similarity threshold
SIMILARITY_THRESHOLD = 0.7
similarity_results = {}
cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

for file, tab_emb in tab_embeddings.items():
    
    tab_feat_names = tab_features_list[file]
    matches_for_this_tab = {}
    for i, time_feat in enumerate(time_features):
        current_time_emb = time_embeddings[i].unsqueeze(0).expand(tab_emb.size(0), -1)
        similarity_scores = cosine_similarity(current_time_emb, tab_emb)

        high_sim_indices = (similarity_scores > SIMILARITY_THRESHOLD).nonzero(as_tuple=True)[0]
        if len(high_sim_indices) == 0:
            matches_for_this_tab[time_feat] = []
            continue
        
        found_matches = []
        for idx in high_sim_indices:
            matched_feature = tab_feat_names[idx.item()]
            score = similarity_scores[idx].item()
            found_matches.append((matched_feature, score))

        found_matches.sort(key=lambda x: x[1], reverse=True)
        
        matches_for_this_tab[time_feat] = found_matches

    similarity_results[file] = matches_for_this_tab

new_dfs = {}

# Extract similar columns from the table
for file, matches in similarity_results.items():
    
    original_df = df_tabs[file]
    columns_to_extract = []
    new_column_names = []

    for base_feat, match_list in matches.items():

        if match_list:
            for i, (matched_col, score) in enumerate(match_list):
                columns_to_extract.append(matched_col)

                if len(match_list) == 1:
                    new_column_names.append(base_feat)
                else:
                    new_column_names.append(f"{base_feat}_{i+1}")

    if columns_to_extract:

        new_df = original_df[columns_to_extract].copy()
        new_df.columns = new_column_names
        new_dfs[file] = new_df

# Save the table data as new DF
df_tab1_relevant = new_dfs['']
df_tab2_relevant = new_dfs['']
df_tab3_relevant = new_dfs['']


#Import Logical Alignment LLM
tokenizer_mnli, model_mnli = load_mnli()
model_mnli.to(device)


id2label = model_mnli.config.id2label
label2id = model_mnli.config.label2id
CONTRADICTION_ID = label2id['CONTRADICTION']

def is_logical_contradiction(feature_base: str, feature_candidate: str, threshold: float = 0.5):
    """
    Use the LLM model to determine whether there is a logical contradiction between two feature names.

    Args:
        feature_base (str): Baseline feature name (e.g., 'is_delayed').
        feature_candidate (str): The feature name to be judged (e.g., 'on_time').
        threshold (float): The probability threshold for determining a contradiction.

    Returns:
        bool: If there is a logical contradiction, return True; otherwise, return False.
    """
    # Construct sentence pairs for NLI tasks
    premise = f"The definition of the data column is '{feature_base}'."
    hypothesis = f"The definition of the data column is '{feature_candidate}'."

    inputs = tokenizer_mnli(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_mnli(**inputs)
    
    logits = outputs.logits
    
    probabilities = torch.softmax(logits, dim=-1)

    contradiction_prob = probabilities[0][CONTRADICTION_ID].item()
    
    return contradiction_prob > threshold

relevant_dfs = {
    'df_tab1_relevant': df_tab1_relevant,
    'df_tab2_relevant': df_tab2_relevant,
    'df_tab3_relevant': df_tab3_relevant
}


processed_dfs = {}

for df_name, df in relevant_dfs.items():

    if df.empty:
        processed_dfs[df_name] = df
        continue

    base_features = df_time.columns.tolist()
    
    columns_to_drop = []
    
    df_processed = df.copy()

    for col_name in df_processed.columns:
        base_feature = col_name.split('_')[0] if '_' in col_name else col_name
        
        if base_feature not in base_features:
            continue

        is_contradiction = is_logical_contradiction(base_feature, col_name)
        
        if is_contradiction:

            if df_processed[col_name].dtype == bool:
                df_processed[col_name] = ~df_processed[col_name] 

            elif pd.api.types.is_numeric_dtype(df_processed[col_name]) and set(df_processed[col_name].unique()).issubset({0, 1}):

                df_processed[col_name] = 1 - df_processed[col_name]
            else:

                columns_to_drop.append(col_name)
        else:
            print(f"[Logical Consistency] Baseline Feature:'{base_feature}' vs. candidate features:'{col_name}'")
    

    if columns_to_drop:
        df_processed.drop(columns=columns_to_drop, inplace=True)

    processed_dfs[df_name] = df_processed

df_tab1_correlation = processed_dfs['df_tab1_relevant']
df_tab2_correlation = processed_dfs['df_tab2_relevant']
df_tab3_correlation = processed_dfs['df_tab3_relevant']

#Import Logical Alignment LLM
tokenizer_unit, model_unit = load_unit()
model_unit.to(device)


#  Predefined unit dictionary (unit: conversion factor), it is recommended to expand it according to the dataset domain.
UNIT_CONVERSION = {
    'weight': {
        'kg': 1.0,
        'g': 0.001,
        'mg': 0.000001,
        'lb': 0.453592,
        'oz': 0.0283495
    },
    'length': {
        'm': 1.0,
        'cm': 0.01,
        'mm': 0.001,
        'km': 1000.0,
        'in': 0.0254,
        'ft': 0.3048,
        'mi': 1609.34
    },
    'time': {
        's': 1.0,
        'ms': 0.001,
        'min': 60.0,
        'h': 3600.0,
        'day': 86400.0
    },
    'volume': {
        'l': 1.0,
        'ml': 0.001,
        'm3': 1000.0,
        'gal': 3.78541,
        'qt': 0.946353,
        'pt': 0.473176
    }
}

# Unit Category Mapping
UNIT_CATEGORIES = {
    'kg': 'weight', 'g': 'weight', 'mg': 'weight', 'lb': 'weight', 'oz': 'weight',
    'm': 'length', 'cm': 'length', 'mm': 'length', 'km': 'length', 'in': 'length',
    'ft': 'length', 'mi': 'length',
    's': 'time', 'ms': 'time', 'min': 'time', 'h': 'time', 'day': 'time',
    'l': 'volume', 'ml': 'volume', 'm3': 'volume', 'gal': 'volume', 'qt': 'volume', 'pt': 'volume'
}


def preprocess_column_name(col_name):

    return str(col_name).lower().replace('_', ' ').strip()


def extract_unit_from_column(col_name):
    """Extract unit information from column names"""
    col_name = str(col_name).lower()
    # #Matches units ending with an underscore
    underscore_match = re.search(r'_([a-z]{1,4})$', col_name)
    if underscore_match:
        return underscore_match.group(1)

    # Matches units within parentheses
    bracket_match = re.search(r'\(([a-z]{1,4})\)$', col_name)
    if bracket_match:
        return bracket_match.group(1)

    # Matches space-separated units
    space_match = re.search(r'\s([a-z]{1,4})$', col_name)
    if space_match:
        return space_match.group(1)

    return None

def detect_unit_category(unit):

    return UNIT_CATEGORIES.get(unit.lower(), None)

def are_units_convertible(unit1, unit2):

    category1 = detect_unit_category(unit1)
    category2 = detect_unit_category(unit2)
    return category1 is not None and category1 == category2

def convert_units(value, from_unit, to_unit):

    category = detect_unit_category(from_unit)
    if category is None or not are_units_convertible(from_unit, to_unit):
        return value

    factor_from = UNIT_CONVERSION[category].get(from_unit.lower(), 1.0)
    factor_to = UNIT_CONVERSION[category].get(to_unit.lower(), 1.0)

    return value * (factor_from / factor_to)

def statistical_unit_check(series1, series2):
    """
    Statistical tests can be used to infer whether two sequences might contain the same physical quantity.
    """

    s1 = series1.dropna()
    s2 = series2.dropna()

    if len(s1) < 10 or len(s2) < 10:
        return False, 1.0

    ratio = (s2.mean() / s1.mean()) if s1.mean() != 0 else 1.0

    min_length = min(len(s1), len(s2))

    s1 = s1.iloc[:min_length]
    s2 = s1.iloc[:min_length]

    slope, intercept, r_value, p_value, std_err = stats.linregress(s1, s2)

    if r_value > 0.9 and abs(intercept) < 0.1 * max(abs(s2.mean()), abs(s1.mean())):
        return True, slope
    return False, 1.0

def find_most_similar_time_col(tab_col, time_cols):

    processed_tab_col = preprocess_column_name(tab_col)
    processed_time_cols = [preprocess_column_name(col) for col in time_cols]

    tab_embedding = model_unit.encode([processed_tab_col])
    time_embeddings = model_unit.encode(processed_time_cols)

    similarities = cosine_similarity(tab_embedding, time_embeddings)[0]
    most_similar_idx = np.argmax(similarities)

    return time_cols[most_similar_idx], similarities[most_similar_idx]

def target_minmax_scale(source_series, target_series):
    """
    Scaling the source sequence to the range of the target sequence
    """
    target_min = target_series.min()
    target_max = target_series.max()
    target_range = target_max - target_min

    source_min = source_series.min()
    source_max = source_series.max()
    source_range = source_max - source_min

    if source_range == 0 or target_range == 0:
        return source_series

    scaled_series = (source_series - source_min) / source_range
    scaled_series = scaled_series * target_range + target_min

    return scaled_series


def align_and_standardize_units(df_time, df_tab_aligned, similarity_threshold=0.6):

    processed_df = df_tab_aligned.copy()
    time_cols = df_time.columns.tolist()

    common_cols = set(processed_df.columns) & set(time_cols)

    for tab_col in df_tab_aligned.columns:
        if tab_col in common_cols:
            time_col = tab_col
            similarity = 1.0
        else:

            time_col, similarity = find_most_similar_time_col(tab_col, time_cols)
            if similarity < similarity_threshold:

                continue

            processed_df = processed_df.rename(columns={tab_col: time_col})

        if not np.issubdtype(processed_df[time_col].dtype, np.number):
            continue

        time_unit = extract_unit_from_column(time_col)
        tab_unit = extract_unit_from_column(tab_col)  

        if time_unit and tab_unit:
            if are_units_convertible(time_unit, tab_unit):

                processed_df[time_col] = processed_df[time_col].apply(
                    lambda x: convert_units(x, tab_unit, time_unit)
                )

            else:
                processed_df[time_col] = target_minmax_scale(
                    processed_df[time_col],
                    df_time[time_col]
                )


        elif time_unit and not tab_unit:
            is_proportional, factor = statistical_unit_check(
                processed_df[time_col], df_time[time_col]
            )

            if is_proportional:
                processed_df[time_col] = processed_df[time_col] * factor

            else:
                processed_df[time_col] = target_minmax_scale(
                    processed_df[time_col],
                    df_time[time_col]
                )

        elif not time_unit and tab_unit:
            continue

        else:

            is_proportional, factor = statistical_unit_check(
                processed_df[time_col], df_time[time_col]
            )

            if is_proportional and abs(factor - 1.0) > 0.01:
                processed_df[time_col] = processed_df[time_col] * factor

            else:
                processed_df[time_col] = target_minmax_scale(
                    processed_df[time_col],
                    df_time[time_col]
                )

    return processed_df

# Execution unit alignment and standardization
df_tab1_final, report1 = align_and_standardize_units(df_time, df_tab1_correlation)
df_tab2_final, report2 = align_and_standardize_units(df_time, df_tab2_correlation)
df_tab3_final, report3 = align_and_standardize_units(df_time, df_tab3_correlation)



def robust_clean_df(df, name):
    df = df.copy().replace([np.inf, -np.inf], np.nan)
    if df.isnull().values.any():
        df = df.fillna(df.mean().fillna(0))
    constant_cols = [col for col in df.columns if df[col].std() <= 1e-8]
    if constant_cols:
        for col in constant_cols:
            df[col] = df[col] + np.random.normal(0, 1e-6, size=len(df))
    return df

print("\n执行表格数据深度清洗...")
scaler_tab = StandardScaler()
tabular_data_list = []
for i, df_raw in enumerate([df_tab1_final, df_tab2_final, df_tab3_final]):
    df_cleaned = robust_clean_df(df_raw, f"Tab {i+1}")
    df_norm = pd.DataFrame(scaler_tab.fit_transform(df_cleaned), columns=df_cleaned.columns)
    tabular_data_list.append(df_norm)

all_features = sorted(list(set().union(*(df.columns for df in tabular_data_list))))
train_days, _ = train_test_split(df_time['Day'].unique(), test_size=0.2, shuffle=False)
train_df = df_time[df_time['Day'].isin(train_days)]
df_time_selected = train_df[all_features]
scaler_time = StandardScaler()
df_time_normalized = pd.DataFrame(scaler_time.fit_transform(df_time_selected), columns=all_features)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TSColumnEncoder(nn.Module):
    def __init__(self, seq_len, d_model=128):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True), num_layers=2)
    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        x = self.pos_encoder(x)
        return self.transformer(x).mean(dim=1)

class SharedTabularProjector(nn.Module):
    def __init__(self, fixed_len, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fixed_len, 256), nn.GELU(), nn.Linear(256, latent_dim))
    def forward(self, x):
        return self.net(x)

class RobustInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, z_ts, z_tab, mask):
        z_ts, z_tab = F.normalize(z_ts, dim=1), F.normalize(z_tab, dim=1)
        logits = torch.matmul(z_ts, z_tab.t()) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stable = logits - logits_max.detach()
        log_prob_denom = torch.log(torch.exp(logits_stable).sum(dim=1) + 1e-8)
        
        valid_indices = torch.where(mask.sum(dim=1) > 0)[0]
        if len(valid_indices) == 0: return torch.tensor(0.0, requires_grad=True).to(z_ts.device)
        
        loss = 0.0
        for i in valid_indices:
            pos_logits = logits_stable[i][mask[i]]
            loss += -(pos_logits - log_prob_denom[i]).mean()
        return loss / len(valid_indices)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_TAB_LEN = 256
latent_dim = 128

filtered_tab_features = []
filtered_tab_names = []

print("\n构建全局特征池...")
for df in tabular_data_list:
    if df.shape[1] < 1: continue
    for col in df.columns:
        col_data = torch.tensor(df[col].values, dtype=torch.float32).view(1, 1, -1)
        col_resized = F.interpolate(col_data, size=FIXED_TAB_LEN, mode='linear', align_corners=False).view(-1)
        filtered_tab_features.append(col_resized)
        filtered_tab_names.append(col)

tab_data_tensor = torch.stack(filtered_tab_features).to(device)
ts_feature_names = list(df_time_normalized.columns)

active_ts_indices = [i for i, name in enumerate(ts_feature_names) if name in filtered_tab_names]
ts_data_tensor = torch.tensor(df_time_normalized.values.T[active_ts_indices], dtype=torch.float32).to(device)
active_ts_names = [ts_feature_names[i] for i in active_ts_indices]

positive_mask = torch.zeros((len(active_ts_names), len(filtered_tab_names)), dtype=torch.bool).to(device)
for i, t_name in enumerate(active_ts_names):
    for j, f_name in enumerate(filtered_tab_names):
        if t_name == f_name: positive_mask[i, j] = True



df_tab1_final1 = pd.DataFrame(scaler_tab.fit_transform(df_tab1_final), columns=df_tab1_final.columns)

df_tab2_final1 = pd.DataFrame(scaler_tab.fit_transform(df_tab2_final), columns=df_tab2_final.columns)

df_tab3_final1 = pd.DataFrame(scaler_tab.fit_transform(df_tab3_final), columns=df_tab3_final.columns)

tabular_data_list = [df_tab1_final1, df_tab2_final1, df_tab3_final1]

global_tab_features = [] 
global_tab_names = []    
global_tab_sources = []  

for t_idx, df_tab in enumerate(tabular_data_list):
    for col in df_tab.columns:
        col_data = torch.tensor(df_tab[col].values, dtype=torch.float32).view(1, 1, -1)
        col_resized = F.interpolate(col_data, size=FIXED_TAB_LEN, mode='linear', align_corners=False)
        
        global_tab_features.append(col_resized.view(-1))
        global_tab_names.append(col)
        global_tab_sources.append(t_idx)


ts_encoder = TSColumnEncoder(seq_len=ts_data_tensor.size(1), d_model=latent_dim).to(device)
tab_projector = SharedTabularProjector(fixed_len=FIXED_TAB_LEN, latent_dim=latent_dim).to(device)
criterion = RobustInfoNCELoss(temperature=0.07).to(device)
optimizer = optim.AdamW(list(ts_encoder.parameters()) + list(tab_projector.parameters()), lr=1e-3)


for epoch in range(50):
    optimizer.zero_grad()
    z_ts = ts_encoder(ts_data_tensor)
    z_tab = tab_projector(tab_data_tensor)
    
    loss = criterion(z_ts, z_tab, positive_mask)
    
    if torch.isnan(loss):
        print(f"NaN at Epoch {epoch}. TS norm: {z_ts.norm().item()}, Tab norm: {z_tab.norm().item()}")
        break
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ts_encoder.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


ts_encoder.eval()
tab_projector.eval()

with torch.no_grad():
    Z_tab_final = tab_projector(tab_data_tensor).cpu()


fused_graph = nx.Graph()
alpha = 0.5 
threshold = 0.1

all_unique_features = sorted(list(set(global_tab_names)))
for node_name in all_unique_features:
    vals = []
    for t_idx, df_tab in enumerate(tabular_data_list):
        if node_name in df_tab.columns:
            vals.append(df_tab[node_name].mean())
    avg_value = np.mean(vals) if vals else 0.0
    fused_graph.add_node(node_name, value=avg_value)



feat_dict = {name: [] for name in all_unique_features}
for idx, name in enumerate(global_tab_names):
    feat_dict[name].append({
        'table_idx': global_tab_sources[idx],
        'embedding': Z_tab_final[idx],
        'raw_data': tabular_data_list[global_tab_sources[idx]][name].values
    })

for i in range(len(all_unique_features)):
    for j in range(i + 1, len(all_unique_features)):
        node_i = all_unique_features[i]
        node_j = all_unique_features[j]
        
        info_i_list = feat_dict[node_i]
        info_j_list = feat_dict[node_j]
        
        emb_i_avg = torch.stack([x['embedding'] for x in info_i_list]).mean(dim=0)
        emb_j_avg = torch.stack([x['embedding'] for x in info_j_list]).mean(dim=0)
        cos_sim = F.cosine_similarity(emb_i_avg.unsqueeze(0), emb_j_avg.unsqueeze(0)).item()
        
        common_tables = set([x['table_idx'] for x in info_i_list]) & set([x['table_idx'] for x in info_j_list])
        
        pearson_list = []
        for t_idx in common_tables:
            raw_i = next(x['raw_data'] for x in info_i_list if x['table_idx'] == t_idx)
            raw_j = next(x['raw_data'] for x in info_j_list if x['table_idx'] == t_idx)
            p_corr = np.corrcoef(raw_i, raw_j)[0, 1]
            if not np.isnan(p_corr):
                pearson_list.append(abs(p_corr))
                
        # 3) 计算最终边权重 (Eq. 14)
        if len(pearson_list) > 0:
            A_base = np.mean(pearson_list)
            final_weight = alpha * A_base + (1 - alpha) * cos_sim
        else:
            final_weight = cos_sim
            

        if final_weight > threshold:
            fused_graph.add_edge(node_i, node_j, weight=final_weight)

config = {
    'batch_size': 16,        
    'hidden_dim': 64,       
    'lr': 5e-4,              
    'weight_decay': 1e-2,    
    'epochs': 100,
    'label_smoothing': 0.1,  
    'dropout': 0.5,          
    'lambda_init': 0.05      
}


torch.manual_seed(42)
np.random.seed(42)

df_time = pd.read_csv("data/Air_time.csv")

def convert_to_timesteps(df, time_col='Time', patient_id_col='Day'):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df['timestep'] = df.groupby(patient_id_col)[time_col].rank(method='first').astype(int) - 1
    df.drop(columns=[time_col], inplace=True)
    return df

def auto_encode_features(df, skip_columns=None, max_unique_for_label=20):
    df = df.copy()
    skip_columns = skip_columns or []
    for col in df.columns:
        if col in skip_columns:
            print(f"jump: {col}")
            continue
        dtype = df[col].dtype
        if dtype == 'bool':
            df[col] = df[col].astype(int)
        elif dtype == 'object' or isinstance(df[col].iloc[0], str):
            num_unique = df[col].nunique()
            if 1 < num_unique <= max_unique_for_label:
                df[col] = df[col].astype('category').cat.codes
            elif num_unique > max_unique_for_label:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)
    return df


df_time = df_time.dropna()
df_time = convert_to_timesteps(df_time)
df_time = auto_encode_features(df_time, skip_columns=['Day'])

def prepare_data(df_time):
    samples = []
    labels = []
    for pid, group in df_time.groupby('Day'):
        time_steps = sorted(group['timestep'].unique())
        seq = []
        for t in time_steps:
            data_t = group[group['timestep'] == t].drop(columns=['Day', 'timestep', 'Target'])
            seq.append(data_t.values)
        sample = np.concatenate(seq, axis=0)
        label = group['Target'].iloc[-1]
        samples.append(sample)
        labels.append(label)
    X = np.stack(samples)
    y = np.array(labels)
    return X, y

X, y = prepare_data(df_time)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
_, y_tensor = torch.unique(y_tensor, return_inverse=True)

graph_node_names = list(fused_graph.nodes()) 
num_nodes = len(graph_node_names) 


ts_feature_names = list(df_time_selected.columns)


active_indices = [ts_feature_names.index(name) for name in graph_node_names]


X_tensor_aligned = X_tensor[:, :, active_indices]


X_node_view = X_tensor_aligned.unsqueeze(-1) 


adj_matrix = nx.to_numpy_array(fused_graph, nodelist=graph_node_names)
static_adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).cuda()

feature_names = list(df_time_selected.columns)

idx_train, idx_val = train_test_split(range(B), test_size=0.2, stratify=y_tensor, random_state=42)
train_loader = DataLoader(TensorDataset(X_node_view[idx_train], y_tensor[idx_train]), 
                          batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(TensorDataset(X_node_view[idx_val], y_tensor[idx_val]), 
                        batch_size=config['batch_size'])


class DGTI_Model(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_classes, static_adj):
        super().__init__()
        self.num_nodes = num_nodes
        self.static_adj = static_adj
        
        self.reg_lambda = nn.Parameter(torch.tensor([config['lambda_init']]))
        self.dynamic_adj_learner = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
        self.gat1 = GATv2Conv(1, hidden_dim // 4, heads=4, dropout=0.2)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=0.2)
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=config['dropout'])
        
        self.time_attention = nn.Linear(hidden_dim, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_seq):
        B, T, N, F_dim = x_seq.shape 
        device = x_seq.device
        
        dyn_adj = F.relu(self.dynamic_adj_learner + self.dynamic_adj_learner.T)
        
        step_representations = []
        gating_values = [] 

        for t in range(T):
            gt = torch.exp(-torch.clamp(self.reg_lambda, min=0.01) * t)
            gating_values.append(gt.item())
            
            fused_adj = gt * self.static_adj + (1.0 - gt) * dyn_adj
            edge_index, _ = dense_to_sparse(fused_adj)
            

            x_t = x_seq[:, t, :, :].reshape(B * N, F_dim)
            h = F.elu(self.gat1(x_t, edge_index))
            h = F.elu(self.gat2(h, edge_index))
            
            h = h.view(B, N, -1)
            step_representations.append(h.mean(dim=1)) # (B, H)
            
        h_seq = torch.stack(step_representations, dim=1)
        gru_out, _ = self.gru(h_seq)
        
        att_scores = F.softmax(self.time_attention(gru_out).squeeze(-1), dim=1)

        final_emb = torch.bmm(att_scores.unsqueeze(1), gru_out).squeeze(1)
        
        return self.classifier(final_emb), gating_values, att_scores
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


num_classes=len(np.unique(y_tensor))


model = DGTI_Model(num_nodes, config['hidden_dim'], num_classes, static_adj_tensor).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
criterion = FocalLoss(alpha=1.5, gamma=2) 


best_val_f1 = 0

for epoch in range(config['epochs']):
    model.train()
    total_loss = 0
    for bx, by in train_loader:
        bx, by = bx.cuda(), by.cuda()
        optimizer.zero_grad()
        
        logits, g_vals, att_weights = model(bx)
        
        loss = criterion(logits, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()

    if epoch % 10 == 0:
        model.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.cuda(), vy.cuda()
                v_logits, v_g, v_att = model(vx)
                v_preds.append(v_logits.argmax(dim=1).cpu())
                v_labels.append(vy.cpu())
        
        y_true = torch.cat(v_labels).numpy()
        y_pred = torch.cat(v_preds).numpy()
        val_f1 = f1_score(y_true, y_pred, average='macro')
        