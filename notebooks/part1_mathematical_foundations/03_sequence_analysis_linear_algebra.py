"""
Sequence Analysis using Linear Algebra

This script demonstrates how linear algebra is applied to biological sequence analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# Set random seed
np.random.seed(42)

def create_dna_sequences(n_sequences=100, sequence_length=200):
    """Create synthetic DNA sequences."""
    nucleotides = ['A', 'C', 'G', 'T']
    sequences = []
    
    for i in range(n_sequences):
        seq = ''.join(np.random.choice(nucleotides, size=sequence_length))
        sequences.append(seq)
    
    return sequences

def encode_dna_sequence(sequence):
    """Encode DNA sequence to one-hot representation."""
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 
                'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded = []
    
    for char in sequence.upper():
        if char in encoding:
            encoded.append(encoding[char])
        else:
            encoded.append([0.25, 0.25, 0.25, 0.25])  # Ambiguous
    
    return np.array(encoded)

def create_protein_sequences(n_sequences=50, sequence_length=300):
    """Create synthetic protein sequences."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequences = []
    
    for i in range(n_sequences):
        seq = ''.join(np.random.choice(list(amino_acids), size=sequence_length))
        sequences.append(seq)
    
    return sequences

def encode_protein_sequence(sequence):
    """Encode protein sequence to one-hot representation."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    encoding = {aa: [1 if i == j else 0 for j in range(20)] 
               for i, aa in enumerate(amino_acids)}
    
    encoded = []
    for char in sequence.upper():
        if char in encoding:
            encoded.append(encoding[char])
        else:
            encoded.append([0] * 20)  # Unknown amino acid
    
    return np.array(encoded)

def calculate_sequence_similarity(sequences, method='hamming'):
    """Calculate similarity matrix between sequences."""
    n_sequences = len(sequences)
    similarity_matrix = np.zeros((n_sequences, n_sequences))
    
    for i in range(n_sequences):
        for j in range(n_sequences):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                if method == 'hamming':
                    # Hamming distance
                    distance = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
                    similarity_matrix[i, j] = 1 - (distance / len(sequences[i]))
                elif method == 'jaccard':
                    # Jaccard similarity
                    set1 = set(sequences[i])
                    set2 = set(sequences[j])
                    similarity_matrix[i, j] = len(set1 & set2) / len(set1 | set2)
    
    return similarity_matrix

def create_kmer_features(sequences, k=3):
    """Create k-mer features from sequences."""
    # Create k-mer vocabulary
    kmers = set()
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmers.add(seq[i:i+k])
    
    # Create feature matrix
    features = []
    for seq in sequences:
        seq_features = {}
        for kmer in kmers:
            seq_features[kmer] = seq.count(kmer)
        features.append(seq_features)
    
    return pd.DataFrame(features)

def analyze_sequence_structure(sequences):
    """Analyze sequence structure using linear algebra."""
    # Create k-mer features
    kmer_features = create_kmer_features(sequences, k=3)
    
    # Apply PCA to k-mer features
    pca = PCA(n_components=min(10, kmer_features.shape[1]))
    kmer_pca = pca.fit_transform(kmer_features)
    
    # Calculate sequence similarity
    similarity_matrix = calculate_sequence_similarity(sequences)
    
    return kmer_features, kmer_pca, similarity_matrix, pca.explained_variance_ratio_

def create_scoring_matrix(amino_acids='ACDEFGHIKLMNPQRSTVWY'):
    """Create a simplified BLOSUM-like scoring matrix."""
    n_aa = len(amino_acids)
    scoring_matrix = np.zeros((n_aa, n_aa))
    
    # Define amino acid properties
    aa_properties = {
        'A': 'hydrophobic', 'V': 'hydrophobic', 'I': 'hydrophobic', 'L': 'hydrophobic',
        'M': 'hydrophobic', 'F': 'hydrophobic', 'W': 'hydrophobic', 'Y': 'hydrophobic',
        'D': 'acidic', 'E': 'acidic',
        'R': 'basic', 'K': 'basic', 'H': 'basic',
        'S': 'polar', 'T': 'polar', 'N': 'polar', 'Q': 'polar',
        'C': 'special', 'G': 'special', 'P': 'special'
    }
    
    # Create scoring matrix
    for i, aa1 in enumerate(amino_acids):
        for j, aa2 in enumerate(amino_acids):
            if i == j:
                scoring_matrix[i, j] = 4  # Match
            elif aa_properties.get(aa1) == aa_properties.get(aa2):
                scoring_matrix[i, j] = 1  # Similar properties
            else:
                scoring_matrix[i, j] = -1  # Different properties
    
    return scoring_matrix, amino_acids

def main():
    """Main function to demonstrate sequence analysis."""
    print("=== Sequence Analysis using Linear Algebra ===\n")
    
    # 1. DNA Sequence Analysis
    print("1. Creating DNA sequences...")
    dna_sequences = create_dna_sequences(n_sequences=50, sequence_length=100)
    print(f"   Created {len(dna_sequences)} DNA sequences of length {len(dna_sequences[0])}")
    
    # Encode DNA sequences
    dna_encoded = np.array([encode_dna_sequence(seq) for seq in dna_sequences])
    print(f"   DNA sequences encoded shape: {dna_encoded.shape}")
    
    # 2. Protein Sequence Analysis
    print("\n2. Creating protein sequences...")
    protein_sequences = create_protein_sequences(n_sequences=30, sequence_length=200)
    print(f"   Created {len(protein_sequences)} protein sequences of length {len(protein_sequences[0])}")
    
    # Encode protein sequences
    protein_encoded = np.array([encode_protein_sequence(seq) for seq in protein_sequences])
    print(f"   Protein sequences encoded shape: {protein_encoded.shape}")
    
    # 3. Sequence Similarity Analysis
    print("\n3. Calculating sequence similarities...")
    dna_similarity = calculate_sequence_similarity(dna_sequences[:20], method='hamming')
    print(f"   DNA similarity matrix shape: {dna_similarity.shape}")
    print(f"   DNA similarity range: [{dna_similarity.min():.3f}, {dna_similarity.max():.3f}]")
    
    # 4. K-mer Feature Analysis
    print("\n4. Creating k-mer features...")
    kmer_features, kmer_pca, similarity_matrix, explained_variance = analyze_sequence_structure(dna_sequences[:20])
    print(f"   K-mer features shape: {kmer_features.shape}")
    print(f"   K-mer PCA shape: {kmer_pca.shape}")
    print(f"   Explained variance by first 5 PCs:")
    for i in range(5):
        print(f"   PC{i+1}: {explained_variance[i]:.3f} ({explained_variance[i]*100:.1f}%)")
    
    # 5. Scoring Matrix
    print("\n5. Creating amino acid scoring matrix...")
    scoring_matrix, aa_list = create_scoring_matrix()
    print(f"   Scoring matrix shape: {scoring_matrix.shape}")
    print(f"   Score range: [{scoring_matrix.min()}, {scoring_matrix.max()}]")
    
    print("\n=== Sequence Analysis Complete ===")

if __name__ == "__main__":
    main() 