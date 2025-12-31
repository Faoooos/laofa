import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from tqdm import tqdm
import torch
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from scipy.stats import wasserstein_distance
import pickle
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, TAGConv
from torch_geometric.utils import dense_to_sparse, from_networkx
from tslearn.metrics import dtw
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns

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

scaler_tab = StandardScaler()
df_tab1_final1 = pd.DataFrame(scaler_tab.fit_transform(df_tab1_final), columns=df_tab1_final.columns)
df_tab2_final1 = pd.DataFrame(scaler_tab.fit_transform(df_tab2_final), columns=df_tab2_final.columns)
df_tab3_final1 = pd.DataFrame(scaler_tab.fit_transform(df_tab3_final), columns=df_tab3_final.columns)
all_features = set(df_tab1_final1.columns) | set(df_tab2_final1.columns) | set(df_tab3_final1.columns)
feature_union = sorted(list(all_features))
df_time_selected = df_time[feature_union]


unique_days = df_time['Day'].unique() 
train_days, test_days = train_test_split(
    unique_days,
    test_size=0.2,  
    shuffle=False,  
    random_state=42  
)
train_df = df_time[df_time['Day'].isin(train_days)]

df_time_selected = train_df[feature_union]

scaler_time = StandardScaler()
df_time_normalized = pd.DataFrame(
    scaler_time.fit_transform(df_time_selected),
    columns=df_time_selected.columns
)

# Transformer position encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, input_dim)
    
    def forward(self, x):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.positional_encoding(x)
        encoded = self.transformer_encoder(x)
        output = self.output_layer(encoded)
        return output
    
# Use Transformer to extract the distribution of each column.
seq_len = 10  
num_samples = len(df_time_normalized) // seq_len
feature_distributions = {}


time_data = df_time_normalized.values[:num_samples*seq_len]
time_data_reshaped = time_data.reshape(num_samples, seq_len, len(feature_union))

transformer = TransformerEncoder(input_dim=len(feature_union))
transformer.eval()

with torch.no_grad():
    tensor_data = torch.tensor(time_data_reshaped, dtype=torch.float32)
    encoded_features = transformer(tensor_data)
    
    for i, feature in enumerate(feature_union):
        feature_data = encoded_features[:, :, i].flatten()
        feature_mean = torch.mean(feature_data).item()
        feature_std = torch.std(feature_data).item()
        
        feature_distributions[feature] = feature_data.cpu().numpy()

# # MMD Loss Calculation
def mmd_loss(x, y, sigma_list=[1, 5, 10]):

    x = x.float()
    y = y.float()
    
    # Calculate the kernel matrix
    def compute_kernel(x, y, sigma):
        # Distance calculation
        xx = torch.cdist(x, x, p=2) ** 2
        yy = torch.cdist(y, y, p=2) ** 2
        xy = torch.cdist(x, y, p=2) ** 2
        
        k_xx = torch.exp(-xx / (2 * sigma ** 2))
        k_yy = torch.exp(-yy / (2 * sigma ** 2))
        k_xy = torch.exp(-xy / (2 * sigma ** 2))
        
        return k_xx, k_yy, k_xy
    
    total_mmd = 0
    for sigma in sigma_list:
        k_xx, k_yy, k_xy = compute_kernel(x, y, sigma)
        mmd = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)
        total_mmd += mmd
    
    return total_mmd / len(sigma_list)

# CGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        
        # Feature extraction section
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        
        # Category Section
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, labels):
        # Feature Extraction
        features = torch.cat([x, labels], dim=1)
        extracted = self.feature_net(features)
        
        # Category score
        logits = self.classifier(extracted)
        
        logits = torch.clamp(logits, min=-10, max=10)
        
        prob = self.sigmoid(logits)
        
        return prob

# CGAN generator
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim, output_dim, hidden_dim=256):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh() 
        )
    
    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.model(x)

# # Define a table dataset class
class TableDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

# Store the reconstruction results and MMD loss for subsequent graph fusion.
reconstructed_tables = []
all_mmd_losses = []

# Train CGAN for each table
for i, (table_name, df_tab) in enumerate([
    ("df_tab1_final1", df_tab1_final1),
    ("df_tab2_final1", df_tab2_final1),
    ("df_tab3_final1", df_tab3_final1)
], 1):
    
    scaler_tab = MinMaxScaler(feature_range=(-1, 1))
    df_tab_normalized = pd.DataFrame(
        scaler_tab.fit_transform(df_tab).astype(np.float32),
        columns=df_tab.columns
    )
    
    # Create a data loader
    dataset = TableDataset(df_tab_normalized.values)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create labels
    labels_mean = torch.tensor(df_tab_normalized.mean(axis=0).values, dtype=torch.float32)
    labels_std = torch.tensor(df_tab_normalized.std(axis=0).values, dtype=torch.float32)
    labels = torch.cat([labels_mean, labels_std], dim=0).unsqueeze(0).repeat(len(df_tab), 1)
    
    latent_dim = 128
    label_dim = labels.shape[1]
    output_dim = len(df_tab.columns)
    
    generator = Generator(latent_dim, label_dim, output_dim)
    discriminator = Discriminator(output_dim, label_dim)
    
    # Optimizer
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion = nn.BCEWithLogitsLoss() 
    
    # MMD loss for each column
    column_mmd_losses = {col: [] for col in df_tab.columns}
    
    epochs = 50
    
    # training
    for epoch in range(epochs):
        epoch_loss_D = 0
        epoch_loss_G = 0
        epoch_mmd = 0
        num_batches = 0
        
        for batch_idx, real_data in enumerate(data_loader):
            batch_size = real_data.size(0)
            num_batches += 1
            
            real_data = real_data.float()
            
            real_labels = torch.ones(batch_size, 1, dtype=torch.float32)
            fake_labels = torch.zeros(batch_size, 1, dtype=torch.float32)
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(labels))
            batch_labels = labels[start_idx:end_idx]
            
            if batch_labels.size(0) != batch_size:
                indices = torch.randint(0, len(labels), (batch_size,))
                batch_labels = labels[indices]
            
            batch_labels = batch_labels.float()
            
            optimizer_D.zero_grad()
            
            output_real_logits = discriminator(real_data, batch_labels)
            loss_real = criterion(output_real_logits, real_labels)
            
            z = torch.randn(batch_size, latent_dim, dtype=torch.float32)
            fake_data = generator(z, batch_labels).detach()
            output_fake_logits = discriminator(fake_data, batch_labels)
            loss_fake = criterion(output_fake_logits, fake_labels)
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            
            optimizer_D.step()
            
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, latent_dim, dtype=torch.float32)
            fake_data = generator(z, batch_labels)
            
            output_fake_logits = discriminator(fake_data, batch_labels)
            loss_G = criterion(output_fake_logits, real_labels)
            
            total_mmd = 0
            valid_columns = 0
            
            for j, col in enumerate(df_tab.columns):
                if col in feature_distributions:
                    fake_col = fake_data[:, j:j+1]
                    real_col = real_data[:, j:j+1]
                    ref_col = torch.tensor(feature_distributions[col], dtype=torch.float32).unsqueeze(1)
                    
                    try:
                        mmd_real = mmd_loss(fake_col, real_col)
                        mmd_ref = mmd_loss(fake_col, ref_col)
                        col_mmd = (mmd_real + mmd_ref) / 2
                        
                        column_mmd_losses[col].append(col_mmd.item())
                        total_mmd += col_mmd
                        valid_columns += 1
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("Warning: Insufficient GPU memory")
                            torch.cuda.empty_cache()
                        else:
                            print(f"MMD calculation error {col}: {str(e)}")
            
            avg_mmd = total_mmd / max(valid_columns, 1)
            
            total_loss_G = loss_G + 0.2 * avg_mmd 
            total_loss_G.backward()
            
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            optimizer_G.step()
            
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            epoch_mmd += avg_mmd.item()
        
        avg_loss_D = epoch_loss_D / num_batches
        avg_loss_G = epoch_loss_G / num_batches
        avg_epoch_mmd = epoch_mmd / num_batches
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss_D: {avg_loss_D:.4f}, '
                  f'Loss_G: {avg_loss_G:.4f}, Avg_MMD: {avg_epoch_mmd:.4f}')
    
    # Calculate the average MMD loss
    avg_column_mmd = {}
    for col in df_tab.columns:
        if column_mmd_losses[col]:
            avg_column_mmd[col] = np.mean(column_mmd_losses[col])
        else:
            avg_column_mmd[col] = 0.0
    
    all_mmd_losses.append(avg_column_mmd)
    
    
    # Generate reconstructed data
    generator.eval()
    with torch.no_grad():
        z = torch.randn(len(df_tab), latent_dim, dtype=torch.float32)
        labels_gen = labels.float()
        generated_data = generator(z, labels_gen)
    
    df_reconstructed = pd.DataFrame(
        scaler_tab.inverse_transform(generated_data.numpy()),
        columns=df_tab.columns
    )
    reconstructed_tables.append(df_reconstructed)

    torch.cuda.empty_cache()

# Construct the Graph Object

graphs = []
table_names = ["df_tab1_final1", "df_tab2_final1", "df_tab3_final1"]

for i, (df_rec, table_name, mmd_losses) in enumerate(zip(reconstructed_tables, table_names, all_mmd_losses)):

    G = nx.Graph()  
    # Add node
    for col in df_rec.columns:
        node_weight = 1.0 / (1.0 + mmd_losses[col])
        G.add_node(col, 
                  value=df_rec[col].mean(), 
                  mmd_loss=mmd_losses[col],
                  weight=node_weight)
    
    # Add edges
    corr_matrix = df_rec.corr()
    for j in range(len(corr_matrix.columns)):
        for k in range(j+1, len(corr_matrix.columns)):
            weight = corr_matrix.iloc[j, k]
            if np.abs(weight) > 0.1:  
                G.add_edge(corr_matrix.columns[j], corr_matrix.columns[k], 
                          weight=weight)
    
    graphs.append(G)


# Create a fusion graph
fused_graph = nx.Graph()

# Collect all unique nodes
all_nodes = set()
for G in graphs:
    all_nodes.update(G.nodes())

# Add node
for node in all_nodes:
    mmds = []
    weights = []
    values = []
    
    for G in graphs:
        if node in G.nodes:
            mmds.append(G.nodes[node]['mmd_loss'])
            weights.append(G.nodes[node]['weight'])
            values.append(G.nodes[node]['value'])
    
    avg_mmd = np.mean(mmds) if mmds else 1.0
    avg_weight = np.mean(weights) if weights else 0.5
    avg_value = np.mean(values) if values else 0.0
    
    fused_graph.add_node(node, 
                        mmd_loss=avg_mmd,
                        weight=avg_weight,
                        value=avg_value)

# Check which subgraphs have edges
print("\nCheck the edge conditions of the subgraph:")
graphs_with_edges = []
for i, G in enumerate(graphs):
    if G.number_of_edges() > 0:
        graphs_with_edges.append((i, G))
        print(f"Subgraph {i+1}: Number of nodes ={G.number_of_nodes()}, Number of edges ={G.number_of_edges()} ✓")
    else:
        print(f"Subgraph {i+1}: Number of nodes ={G.number_of_nodes()}, Number of edges ={G.number_of_edges()} ✗ ((Will be ignored)")

# Collect edge information only from subgraphs with edges
edge_weights = {}
edge_counts = {}

for graph_idx, G in graphs_with_edges:
    for edge in G.edges():
        edge = tuple(sorted(edge))
        weight = G[edge[0]][edge[1]]['weight']
        edge_weights[edge] = edge_weights.get(edge, 0) + weight
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

# Add edge to fusion graph

for edge in edge_weights:
    avg_weight = edge_weights[edge] / edge_counts[edge]
    

    if edge[0] in fused_graph.nodes and edge[1] in fused_graph.nodes:
        node1_weight = fused_graph.nodes[edge[0]].get('weight', 0.5)
        node2_weight = fused_graph.nodes[edge[1]].get('weight', 0.5)
        
        adjusted_weight = avg_weight * (node1_weight + node2_weight) / 2
        
        if np.isnan(adjusted_weight) or np.isinf(adjusted_weight):
            adjusted_weight = avg_weight 
        
        fused_graph.add_edge(edge[0], edge[1], weight=adjusted_weight)
    else:
        print(f"Warning: The node for edge {edge} does not exist. Skip.")

print(f"\n Statistics of fused graphs:")
print(f"Number of nodes: {fused_graph.number_of_nodes()}")
print(f"Number of edges: {fused_graph.number_of_edges()}")
print(f"Number of subgraphs used: {len(graphs_with_edges)}/{len(graphs)}")


torch.manual_seed(42)
np.random.seed(42)

# Preprocessing
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

# Standardization
scaler = StandardScaler()
X_scaled = X.copy()
for i in range(X.shape[0]):
    X_scaled[i] = scaler.fit_transform(X[i])

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

class_counts = np.bincount(y_tensor.numpy())
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * len(class_counts)


if len(X_tensor.shape) == 3:
    B, TN, F = X_tensor.shape
    # Adjust according to your actual data
    N = 1  
    T = TN // N
    X_tensor = X_tensor.view(B, T, N, F)


class LearnableGraph(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.learnable_adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
    def forward(self, x):

        adj = self.learnable_adj
        adj = adj + adj.T 
        adj = adj - torch.diag(torch.diag(adj))
        adj = nn.functional.gelu(adj)
        adj = nn.functional.normalize(adj, p=1, dim=1)
        return adj

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.att = nn.Linear(hidden_dim, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        x: (B, T, H)
        """
        weights = nn.functional.softmax(self.att(x).squeeze(-1), dim=1) 
        weighted = torch.bmm(weights.unsqueeze(1), x).squeeze(1) 
        
        weighted = self.layer_norm(weighted + x.mean(dim=1))
        return weighted

class GraphTimeModel(nn.Module):
    def __init__(self, num_features, num_classes, num_nodes=13,
                 hidden_dim=128, heads=4, static_edge_index=None,
                 fusion_strategy='adaptive'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy
        self.static_edge_index = static_edge_index

        self.graph_learner = LearnableGraph(num_nodes, hidden_dim)

        self.decay_base = nn.Parameter(torch.tensor(0.5))
        self.decay_rate = nn.Parameter(torch.tensor(0.1))
        self.decay_type = 'exponential'

        self.gat1 = GATConv(num_features, hidden_dim, heads=heads, concat=False)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.attention = SelfAttention(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def compute_time_decay(self, current_time, total_times, node_importance=None):
        """
        Calculate time decay weight
        """
        t_normalized = current_time / max(total_times - 1, 1)
        
        if self.decay_type == 'exponential':
            base_decay = torch.sigmoid(self.decay_base)
            rate_factor = torch.sigmoid(self.decay_rate)
            decay = base_decay * torch.exp(-rate_factor * t_normalized)
            
        elif self.decay_type == 'linear':
            decay = 1.0 - t_normalized
            
        elif self.decay_type == 'sigmoid':
            midpoint = 0.5
            steepness = 10
            decay = 1 / (1 + torch.exp(-steepness * (t_normalized - midpoint)))
            
        else:
            decay = torch.exp(-t_normalized)
        
        # # Slower decay of important nodes
        if node_importance is not None:
            importance_factor = 1.0 - node_importance
            decay = decay * importance_factor
            
        return decay.clamp(min=0.1, max=1.0)
    
    def compute_dynamic_fusion_weights(self, static_edge_weights, dynamic_edge_weights, 
                                   time_decay_weights, mmd_weights):
        """
        static_edge_weights: Static graph edge weights
        dynamic_edge_weights: Dynamic graph edge weights  
        time_decay_weights: Time decay weight
        """
        if self.fusion_strategy == 'adaptive':

            fusion_alpha = torch.sigmoid(torch.mean(mmd_weights))
            
            dynamic_weights = time_decay_weights * (
                fusion_alpha * static_edge_weights + 
                (1 - fusion_alpha) * dynamic_edge_weights
            )
            
        elif self.fusion_strategy == 'decay':
            dynamic_weights = time_decay_weights * static_edge_weights
            
        else:
            dynamic_weights = time_decay_weights * static_edge_weights
            
        return dynamic_weights

    def forward(self, x_seq, edge_index_static):
        """
        x_seq: shape (B, T, N, F)
        edge_index_static: shape (2, E)
        """
        B, T, N, F = x_seq.shape
        device = x_seq.device

        outputs = []
        all_time_decay_weights = []

        for t in range(T):
            x_t = x_seq[:, t, :, :]  # shape: (B, N, F)

            adj = self.graph_learner(x_t)  # shape: (N, N)
            edge_index, edge_weight = dense_to_sparse(adj)

            time_decay_weights_t = self.compute_time_decay(t, T)
            all_time_decay_weights.append(time_decay_weights_t)

            # # Calculate the fusion weight for each edge
            if edge_index_static.numel() > 0:

                static_edge_weights_list = []
                
                static_edge_weights_tensor = torch.tensor(static_edge_weights_list, device=device)
            else:
                static_edge_weights_tensor = None

            if edge_index.numel() > 0 and static_edge_weights_tensor is not None:
                dynamic_fusion_weights = self.compute_dynamic_fusion_weights(
                    static_edge_weights_tensor, 
                    edge_weight, 
                    time_decay_weights_t.unsqueeze(0),
                )
            else:
                dynamic_fusion_weights = edge_weight


            edge_index_combined = torch.cat([edge_index_static.to(device), edge_index.to(device)], dim=1)

            x_flat = x_t.contiguous().view(B * N, F) 
            
            h = self.gat1(x_flat, edge_index_combined, edge_attr=dynamic_fusion_weights)
            h = nn.functional.gelu(h)           
            h = h.contiguous().view(B, N, -1)
            
            h_flat = h.contiguous().view(B * N, -1)
            h = self.gcn1(h_flat, edge_index_combined, edge_attr=dynamic_fusion_weights)
            h = nn.functional.gelu(h)
            h = h.contiguous().view(B, N, -1)

            h_flat = h.contiguous().view(B * N, -1)
            h = self.gcn1(h_flat, edge_index_combined, edge_attr=dynamic_fusion_weights)
            h = nn.functional.gelu(h)
            h = h.contiguous().view(B, N, -1)

            h_pooled = h.mean(dim=1)  
            outputs.append(h_pooled)

        h_seq = torch.stack(outputs, dim=1)  # shape: (B, T, H)
        
        gru_out, _ = self.gru(h_seq)

        att_out = self.attention(gru_out)

        final_representation = self.fusion_gate(torch.cat([att_out, gru_out], dim=-1))

        logits = self.classifier(final_representation)

        return logits
    

try:
    data_static = from_networkx(fused_graph)
    edge_index_static = data_static.edge_index
except:
    print("Warning: No fusion graph found, creating empty edge indexes.")
    edge_index_static = torch.empty((2, 0), dtype=torch.long)

B, T, N, F = X_tensor.shape

model = GraphTimeModel(
    num_features=F,
    num_classes=len(np.unique(y_tensor)),
    num_nodes=N,
    hidden_dim=128,
    heads=4,
    static_edge_index=edge_index_static,
    fusion_strategy='adaptive' 
)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.001, 
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10, 
    T_mult=2, 
    eta_min=1e-6
)


print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")

train_idx, val_idx = train_test_split(
    range(len(X_tensor)), 
    test_size=0.2, 
    stratify=y_tensor.numpy(),
    random_state=42
)

X_train = X_tensor[train_idx]
y_train = y_tensor[train_idx]
X_val = X_tensor[val_idx]
y_val = y_tensor[val_idx]


train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None

for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        
        output = model(X_train, edge_index_static)
        loss = criterion(output, y_train)
        
        l2_reg = torch.tensor(0., device=output.device)
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        loss += 1e-5 * l2_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_output = model(X_val, edge_index_static)
            val_loss = criterion(val_output, y_val)
            val_losses.append(val_loss.item())
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0:
            train_pred = output.argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean()
            val_pred = val_output.argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean()
            

model.eval()

# evaluate
with torch.no_grad():
    val_logits, _ = model(X_val, edge_index_static)
    val_probs = nn.functional.softmax(val_logits, dim=1)
    val_pred = val_logits.argmax(dim=1)
    
    val_true = y_val.numpy()
    val_pred_np = val_pred.numpy()
    val_probs_np = val_probs.numpy()

    f1_macro = f1_score(val_true, val_pred_np, average='macro')
    f1_weighted = f1_score(val_true, val_pred_np, average='weighted')

    n_classes = len(np.unique(val_true))

    auroc = roc_auc_score(val_true, val_probs_np[:, 1])
    auprc = average_precision_score(val_true, val_probs_np[:, 1])
