# Required imports
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------ First Clustering Stage ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Simple needed functions

def remove_word_after_busybox(command):
    pattern = r'(/bin/busybox)\s+\w+'
    return re.sub(pattern, r'\1', command)

def replace_root(command):
    pattern = r'root:(\w+)'
    return re.sub(pattern, r'root', command)

def replace_words(command):
    pattern = r'"\w+\\\\n\w+"'
    return re.sub(pattern, 'replace', command)

def calculate_length(command):
    return len(command)
 
def calculate_count(command):
    words = command.split()
    return len(words)

def extract_password(name):
    name_list = eval(name)  # Convert the string to a list
    if len(name_list) >= 2:
        return name_list[1]
    else:
        return ''

def remove_word_after_echo(command):
    pattern = r'echo \w{16}'
    return re.sub(pattern, 'value', command)

def replace_root_without_dots(command):
    pattern = r'"root [^\s]+"'
    return re.sub(pattern, 'password', command)

    

# Command normalization--> used to reduce dimensionality of unique commands
def command_normalization(df):

    # Remove the brackets [ ]
    df['_source.commands'] = df['_source.commands'].str.replace(r'\[|\]', '', regex=True)

    # Remove any characters after "/bin/busybox"
    # Command example:
    '''
    /bin/busybox ZSQCG --> /bin/busybox
    /bin/busybox FIGKX --> /bin/busybox
    '''
    df['_source.commands'] = df['_source.commands'].apply(lambda x: remove_word_after_busybox(x))

    # Replace 'root: password' by a master word
    # Command example
    '''
    'echo "root:Q9szwntZCIbJ"|chpasswd|bash --> 'echo "root:root"|chpasswd|bash
    'echo "root:2HPVq2TaCRqY"|chpasswd|bash' -->  'echo "root:root"|chpasswd|bash
    '''
    df['_source.commands'] = df['_source.commands'].apply(lambda x: replace_root(x))

    # Replace word1\\\\nword2 to a general form
    # Command example
    '''
    "W9xs8uuSYZPN\\\\nW9xs8uuSYZPN" --> replace
    Q54CTkBEzSCe\\\\nQ54CTkBEzSCe --> replace
    '''
    df['_source.commands'] = df['_source.commands'].apply(lambda x: replace_words(x))

    # Remove empty string
    pattern = r"^''+$"
    df = df[~df['_source.commands'].str.match(pattern)]

    # Remove command with just "'w'"
    df = df[~df['_source.commands'].str.strip().eq("'w'")]
    return df

# Laplacian computation
def compute_Lrw(cosine_mat):
    # Laplacian Matrix
    # Step 1: Calculate the degree matrix D
    D = np.diag(np.sum(cosine_mat, axis=1))

    # Step 2: Calculate the degree-normalized Laplacian matrix (L_rw)
    D_inv = np.linalg.inv(D)
    L_rw = np.identity(cosine_mat.shape[0]) - np.dot(D_inv, cosine_mat)

    return L_rw

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------ Second Clustering Stage ---------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def feature_extraction(df):
    # Attack duration
    df_features = pd.DataFrame()
    df_features['attack_duration'] = (pd.to_datetime(df['_source.endTime']) - pd.to_datetime(df['_source.startTime'])).dt.total_seconds()

    # Length and the word count
    df_features['length_command'] = df['_source.commands'].apply(lambda x: calculate_length(x))
    df_features['wcount_command'] = df['_source.commands'].apply(lambda x: calculate_count(x))

    # Download link, yes or no
    # Define a regular expression to match "wget" and a URL
    regex_pattern = r'\bhttps?://\S+\b'
    # Use str.contains to identify rows matching the pattern
    df_features['link_download'] = df['_source.commands'].str.contains(regex_pattern, case=False, regex=True)
    df_features['link_download'] = df_features['link_download'].apply(lambda x: 'yes' if x else 'no')

    # Match chmod 777
    regex_pattern = r'\bchmod 777\b'
    df_features['chmod_found'] = df['_source.commands'].str.contains(regex_pattern, case=False, regex=True)
    df_features['chmod_found'] = df_features['chmod_found'].apply(lambda x: 'yes' if x else 'no')

    # Length of password
    # Extract password
    passwords = df['_source.loggedin'].apply(lambda x: extract_password(x))
    # Compute length
    df_features['length_password'] = passwords.apply(lambda x: calculate_length(x))

    # Protocol 
    df_features['protocol'] = df['_source.protocol']

    # Host and peer port
    df_features['host_port'] = df['_source.hostPort']
    df_features['peer_port'] = df['_source.peerPort']

    # Continent of the attacker
    df_features['continent_attacker'] = df['_source.geoip.continent_code']
    # Fill NAN with NA, as seen in EDA
    df_features['continent_attacker'] = df_features['continent_attacker'].fillna('NA')

    # Add labels of clusters
    df_features['spectral_clustering'] = df['spectral_cluster']
    return df_features

def plot_corr_mat(df):

    # Correlation matrix
    df_corr = df.copy()
    df_corr = df_corr.drop(['continent_attacker'],axis=1)

    # Encode binary features
    le = LabelEncoder()
    df_corr['link_download'] = le.fit_transform(df['link_download'])
    df_corr['chmod_found'] = le.fit_transform(df['chmod_found'])
    df_corr['protocol'] = le.fit_transform(df['protocol'])

    # Correlation plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', linewidths=0.1)
    plt.title('Correlation matrix')
    plt.show()
    del df_corr

def cluster_visualization(df, cluster_name):
    # Clustering visualization

    fig, ax = plt.subplots(3, 2, figsize=(12, 10))

    # Mean attack duration per cluster
    grouped = df.groupby(cluster_name)['attack_duration'].mean()
    clusters = grouped.index
    mean_duration = grouped.values
    ax[0,0].bar(clusters, mean_duration, color='skyblue', edgecolor='navy', alpha=0.7)
    ax[0,0].set_xlabel('Cluster', fontsize=14)
    ax[0,0].set_ylabel('Mean attack duration (s)', fontsize=14)
    ax[0,0].set_title('Mean attack duration by cluster', fontsize=16, fontweight='bold')
    ax[0,0].grid(axis='y', linestyle='--', alpha=0.6)
    ax[0,0].set_xticks(clusters)

    # Mean word count command per cluster
    grouped = df.groupby(cluster_name)['wcount_command'].mean()
    clusters = grouped.index
    mean_wcount = grouped.values
    ax[0,1].bar(clusters, mean_wcount, color='skyblue', edgecolor='navy', alpha=0.7)
    ax[0,1].set_xlabel('Cluster', fontsize=14)
    ax[0,1].set_ylabel('Mean word count command', fontsize=14)
    ax[0,1].set_title('Mean word count command by cluster', fontsize=16, fontweight='bold')
    ax[0,1].grid(axis='y', linestyle='--', alpha=0.6)
    ax[0,1].set_xticks(clusters)

    # Mean length password per cluster
    grouped = df.groupby(cluster_name)['length_password'].mean()
    clusters = grouped.index
    mean_password = grouped.values
    ax[1,0].bar(clusters, mean_password, color='skyblue', edgecolor='navy', alpha=0.7)
    ax[1,0].set_xlabel('Cluster', fontsize=14)
    ax[1,0].set_ylabel('Mean length password', fontsize=14)
    ax[1,0].set_title('Mean length password by cluster', fontsize=16, fontweight='bold')
    ax[1,0].grid(axis='y', linestyle='--', alpha=0.6)
    ax[1,0].set_xticks(clusters)

    # Link downloaded per cluster
    grouped = df.groupby([cluster_name, 'link_download']).size().unstack(fill_value=0)
    clusters = grouped.index
    count_yes = grouped['yes']
    count_no = grouped['no']
    width = 0.4
    ax[1,1].bar(clusters - width/2, count_yes, width, color='skyblue', edgecolor='navy', alpha=0.7, label='Yes')
    ax[1,1].bar(clusters + width/2, count_no, width, color='orange', edgecolor='maroon', alpha=0.7, label='No')
    ax[1,1].set_xlabel('Cluster', fontsize=14)
    ax[1,1].set_ylabel('Count', fontsize=14)
    ax[1,1].set_title('Link download in Each Cluster', fontsize=16, fontweight='bold')
    ax[1,1].grid(axis='y', linestyle='--', alpha=0.6)
    ax[1,1].set_xticks(clusters)
    ax[1,1].legend()

    # Protocol used per cluster
    grouped = df.groupby([cluster_name, 'protocol']).size().unstack(fill_value=0)
    clusters = grouped.index
    count_ssh = grouped['ssh']
    count_telnet = grouped['telnet']
    width = 0.4
    ax[2,0].bar(clusters - width/2, count_ssh, width, color='skyblue', edgecolor='navy', alpha=0.7, label='ssh')
    ax[2,0].bar(clusters + width/2, count_telnet, width, color='orange', edgecolor='maroon', alpha=0.7, label='telnet')
    ax[2,0].set_xlabel('Cluster', fontsize=14)
    ax[2,0].set_ylabel('Count', fontsize=14)
    ax[2,0].set_title('Protocol used by Cluster', fontsize=16, fontweight='bold')
    ax[2,0].grid(axis='y', linestyle='--', alpha=0.6)
    ax[2,0].set_xticks(clusters)
    ax[2,0].legend()

    # chmod commandfound per cluster
    grouped = df.groupby([cluster_name, 'chmod_found']).size().unstack(fill_value=0)
    clusters = grouped.index
    count_yes = grouped['yes']
    count_no = grouped['no']
    width = 0.4
    ax[2,1].bar(clusters - width/2, count_yes, width, color='skyblue', edgecolor='navy', alpha=0.7, label='Yes')
    ax[2,1].bar(clusters + width/2, count_no, width, color='orange', edgecolor='maroon', alpha=0.7, label='No')
    ax[2,1].set_xlabel('Cluster', fontsize=14)
    ax[2,1].set_ylabel('Count', fontsize=14)
    ax[2,1].set_title('Chmod found per cluster', fontsize=16, fontweight='bold')
    ax[2,1].grid(axis='y', linestyle='--', alpha=0.6)
    ax[2,1].set_xticks(clusters)
    ax[2,1].legend()


    # Show the plot
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------ Third Clustering Stage ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def command_improve_normalization(df):
    df = command_normalization(df)

    # Replace echo 23497y92374y324 to a general form
    # Command example
    '''
    echo 23497y92374y324--> echo value
    echo dcj38NS8n208311--> echo value
    '''
    df['_source.commands'] = df['_source.commands'].apply(lambda x: remove_word_after_echo(x))

    # Replace root password to a general form
    # Command example
    '''
    root admin-->echo password
    echo Apple2--> echo password
    '''
    df['_source.commands'] = df['_source.commands'].apply(lambda x: replace_root_without_dots(x))
    return df

def train(X_train,X_test,y_train,y_test):

    # Define a list of models
    models = [
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier(random_state=42, class_weight='balanced')
        },
        {
            'name': 'SVM',
            'model': SVC(probability=False, random_state=42, class_weight='balanced')
        }
    ]

    # Define hyperparameter grids for RandomForest
    rf_param_grid = {
        'model__n_estimators': [50, 100, 150],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    # Define hyperparameter grids for SVM
    svm_param_grid = {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto', 0.1, 1]
    }

    # Create a dictionary to map models to their respective parameter grids
    param_grids = {
        'RandomForest': rf_param_grid,
        'SVM': svm_param_grid
    }

    # Create an empty list to store the best models and results
    best_models = []

    # Iterate over the models and perform hyperparameter tuning
    for model_info in models:
        model_name = model_info['name']
        model = model_info['model']

        # Define our Pipeline
        steps = [
            ('col_selector', ColumnSelector(cols=('_source.commands'), drop_axis=True)), # Select column
            ('tfidf', TfidfVectorizer()), # Perform Tf idf
            ('model', model)     # Model step
        ]

        # Create the pipeline
        pipeline = Pipeline(steps)

        # Define the hyperparameter grid for the current model
        param_grid = param_grids[model_name]

        # Perform GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Make predictions with the best model
        y_pred = best_model.predict(X_test)

        # Calculate the F1 score
        f1 = f1_score(y_test, y_pred, average='macro')

        # Store the best model and its results
        best_models.append({'model_name': model_name, 'best_model': best_model, 'f1_score': f1,'output':y_pred})
        
    return best_models

