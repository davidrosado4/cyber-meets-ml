# Required imports
import re
import pandas as pd
import numpy as np

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

