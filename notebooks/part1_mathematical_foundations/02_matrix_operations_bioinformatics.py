"""
Matrix Operations in Bioinformatics

This script demonstrates key matrix operations and their biological applications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed
np.random.seed(42)

def create_gene_expression_data():
    """Create synthetic gene expression data."""
    n_samples = 60  # 30 tumor + 30 control
    n_genes = 15000
    
    # Create expression matrix
    expression_matrix = np.random.lognormal(mean=5, sigma=1, size=(n_samples, n_genes))
    
    # Add biological structure
    tumor_genes = np.random.choice(n_genes, size=100, replace=False)
    control_genes = np.random.choice(n_genes, size=50, replace=False)
    
    # Differential expression
    expression_matrix[:30, tumor_genes] *= 2.5  # Upregulated in tumor
    expression_matrix[30:, control_genes] *= 1.8  # Upregulated in control
    
    return expression_matrix, tumor_genes, control_genes

def calculate_gene_correlations(expression_matrix):
    """Calculate correlation matrix between genes."""
    # Standardize the data
    expression_std = (expression_matrix - expression_matrix.mean(axis=0)) / expression_matrix.std(axis=0)
    
    # Calculate correlation matrix
    n_genes = expression_matrix.shape[1]
    correlation_matrix = np.zeros((n_genes, n_genes))
    
    for i in range(n_genes):
        for j in range(n_genes):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                numerator = np.sum(expression_std[:, i] * expression_std[:, j])
                denominator = np.sqrt(np.sum(expression_std[:, i]**2) * np.sum(expression_std[:, j]**2))
                correlation_matrix[i, j] = numerator / denominator if denominator != 0 else 0
    
    return correlation_matrix

def svd_from_scratch(X, n_components=None):
    """Implement SVD from scratch using NumPy."""
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # Compute covariance matrix
    cov_matrix = X_centered.T @ X_centered
    
    # Eigenvalue decomposition
    eigenvals, V = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    V = V[:, idx]
    
    # Compute singular values and U
    S = np.sqrt(eigenvals)
    U = X_centered @ V / S
    
    # Handle zero singular values
    mask = S > 1e-10
    S = S[mask]
    U = U[:, mask]
    Vt = V[:, mask].T
    
    return U, S, Vt

def pca_from_scratch(X, n_components=None):
    """Implement PCA from scratch using SVD."""
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # Apply SVD
    U, S, Vt = svd_from_scratch(X_centered, n_components)
    
    # Transform data
    X_pca = U @ np.diag(S)
    
    # Calculate explained variance
    total_variance = np.sum(X_centered**2)
    explained_variance = S**2 / total_variance
    
    return X_pca, Vt, explained_variance

def create_protein_interaction_matrix(n_proteins=50):
    """Create synthetic protein-protein interaction matrix."""
    # Create symmetric matrix with some structure
    interaction_matrix = np.random.random((n_proteins, n_proteins))
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2  # Make symmetric
    
    # Add some protein complexes (clusters)
    complex1 = np.random.choice(n_proteins, size=8, replace=False)
    complex2 = np.random.choice(n_proteins, size=6, replace=False)
    
    # Increase interactions within complexes
    for i in complex1:
        for j in complex1:
            if i != j:
                interaction_matrix[i, j] += 0.3
    
    for i in complex2:
        for j in complex2:
            if i != j:
                interaction_matrix[i, j] += 0.4
    
    # Ensure values are between 0 and 1
    interaction_matrix = np.clip(interaction_matrix, 0, 1)
    
    return interaction_matrix

def analyze_protein_network(interaction_matrix):
    """Analyze protein interaction network properties."""
    n_proteins = interaction_matrix.shape[0]
    
    # Calculate degree centrality
    degree_centrality = np.sum(interaction_matrix > 0.5, axis=1)
    
    # Calculate clustering coefficient
    clustering_coeff = []
    for i in range(n_proteins):
        neighbors = np.where(interaction_matrix[i, :] > 0.5)[0]
        if len(neighbors) < 2:
            clustering_coeff.append(0)
        else:
            # Calculate triangles
            triangles = 0
            for j in neighbors:
                for k in neighbors:
                    if j < k and interaction_matrix[j, k] > 0.5:
                        triangles += 1
            
            max_triangles = len(neighbors) * (len(neighbors) - 1) / 2
            clustering_coeff.append(triangles / max_triangles if max_triangles > 0 else 0)
    
    return degree_centrality, np.array(clustering_coeff)

def main():
    """Main function to demonstrate matrix operations."""
    print("=== Matrix Operations in Bioinformatics ===\n")
    
    # 1. Gene Expression Data
    print("1. Creating gene expression data...")
    expression_matrix, tumor_genes, control_genes = create_gene_expression_data()
    print(f"   Expression matrix shape: {expression_matrix.shape}")
    
    # 2. Gene Correlation Analysis
    print("\n2. Calculating gene correlations...")
    subset_size = 100
    expression_subset = expression_matrix[:, :subset_size]
    correlation_matrix = calculate_gene_correlations(expression_subset)
    print(f"   Correlation matrix shape: {correlation_matrix.shape}")
    print(f"   Correlation range: [{correlation_matrix.min():.3f}, {correlation_matrix.max():.3f}]")
    
    # 3. SVD Implementation
    print("\n3. Applying SVD...")
    U, S, Vt = svd_from_scratch(expression_matrix, n_components=10)
    print(f"   U shape: {U.shape}")
    print(f"   S shape: {S.shape}")
    print(f"   Vt shape: {Vt.shape}")
    print(f"   First 5 singular values: {S[:5]}")
    
    # 4. PCA Implementation
    print("\n4. Applying PCA...")
    X_pca, components, explained_variance = pca_from_scratch(expression_matrix, n_components=10)
    print(f"   PCA transformed shape: {X_pca.shape}")
    print(f"   Explained variance by first 5 PCs:")
    for i in range(5):
        print(f"   PC{i+1}: {explained_variance[i]:.3f} ({explained_variance[i]*100:.1f}%)")
    
    # 5. Protein Interaction Network
    print("\n5. Creating protein interaction network...")
    ppi_matrix = create_protein_interaction_matrix()
    print(f"   PPI matrix shape: {ppi_matrix.shape}")
    
    # 6. Network Analysis
    print("\n6. Analyzing protein network...")
    degree_cent, clustering_coeff = analyze_protein_network(ppi_matrix)
    print(f"   Average degree centrality: {degree_cent.mean():.2f}")
    print(f"   Average clustering coefficient: {clustering_coeff.mean():.3f}")
    
    print("\n=== Matrix Operations Complete ===")

if __name__ == "__main__":
    main() 