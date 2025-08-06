"""
Linear Algebra Exercises for Bioinformatics

This file contains exercises to practice linear algebra concepts in bioinformatics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed
np.random.seed(42)

class LinearAlgebraExercises:
    """Collection of linear algebra exercises for bioinformatics."""
    
    def __init__(self):
        """Initialize the exercises."""
        self.data = None
        self.results = {}
    
    def exercise_1_svd_implementation(self):
        """
        Exercise 1: Implement SVD from scratch
        
        Task: Implement Singular Value Decomposition without using numpy.linalg.svd
        """
        print("=== Exercise 1: SVD Implementation ===")
        
        # Create test data
        X = np.random.randn(20, 10)
        
        # TODO: Implement SVD from scratch
        # Hint: Use eigenvalue decomposition of X^T X
        
        # Your implementation should return U, S, Vt
        # U: Left singular vectors
        # S: Singular values
        # Vt: Right singular vectors (transposed)
        
        # Expected output format:
        # U, S, Vt = your_svd_implementation(X)
        
        print("TODO: Implement SVD from scratch")
        print("Expected: U shape (20, rank), S shape (rank,), Vt shape (rank, 10)")
        
        return None, None, None
    
    def exercise_2_pca_implementation(self):
        """
        Exercise 2: Implement PCA from scratch
        
        Task: Implement Principal Component Analysis using your SVD implementation
        """
        print("\n=== Exercise 2: PCA Implementation ===")
        
        # Create test data
        X = np.random.randn(50, 20)
        
        # TODO: Implement PCA using SVD
        # Steps:
        # 1. Center the data (subtract mean)
        # 2. Apply SVD to centered data
        # 3. Transform data using U * S
        # 4. Calculate explained variance
        
        # Expected output:
        # X_pca, components, explained_variance = your_pca_implementation(X, n_components=5)
        
        print("TODO: Implement PCA from scratch")
        print("Expected: X_pca shape (50, 5), components shape (5, 20), explained_variance shape (5,)")
        
        return None, None, None
    
    def exercise_3_gene_expression_analysis(self):
        """
        Exercise 3: Gene Expression Analysis
        
        Task: Analyze gene expression data using PCA and interpret results
        """
        print("\n=== Exercise 3: Gene Expression Analysis ===")
        
        # Create synthetic gene expression data
        n_samples = 40  # 20 tumor + 20 control
        n_genes = 1000
        
        # Create expression matrix with structure
        expression_matrix = np.random.lognormal(mean=5, sigma=1, size=(n_samples, n_genes))
        
        # Add differential expression
        tumor_genes = np.random.choice(n_genes, size=50, replace=False)
        expression_matrix[:20, tumor_genes] *= 2.0  # Upregulated in tumor
        
        # TODO: Analyze this data
        # 1. Apply PCA to the expression matrix
        # 2. Visualize the first 2 principal components
        # 3. Identify which genes contribute most to PC1
        # 4. Check if tumor-specific genes are overrepresented in top PC1 loadings
        
        print("TODO: Analyze gene expression data")
        print("1. Apply PCA")
        print("2. Create scatter plot of PC1 vs PC2")
        print("3. Find top genes contributing to PC1")
        print("4. Check overlap with tumor-specific genes")
        
        return expression_matrix, tumor_genes
    
    def exercise_4_protein_network_analysis(self):
        """
        Exercise 4: Protein Interaction Network Analysis
        
        Task: Analyze protein interaction network using matrix operations
        """
        print("\n=== Exercise 4: Protein Network Analysis ===")
        
        # Create protein interaction matrix
        n_proteins = 30
        interaction_matrix = np.random.random((n_proteins, n_proteins))
        interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2  # Make symmetric
        
        # Add protein complexes
        complex1 = np.random.choice(n_proteins, size=5, replace=False)
        complex2 = np.random.choice(n_proteins, size=4, replace=False)
        
        for i in complex1:
            for j in complex1:
                if i != j:
                    interaction_matrix[i, j] += 0.5
        
        for i in complex2:
            for j in complex2:
                if i != j:
                    interaction_matrix[i, j] += 0.6
        
        interaction_matrix = np.clip(interaction_matrix, 0, 1)
        
        # TODO: Analyze the network
        # 1. Calculate degree centrality for each protein
        # 2. Find the most connected proteins (hubs)
        # 3. Calculate clustering coefficient for each protein
        # 4. Identify protein complexes using clustering
        
        print("TODO: Analyze protein interaction network")
        print("1. Calculate degree centrality")
        print("2. Find protein hubs")
        print("3. Calculate clustering coefficients")
        print("4. Identify protein complexes")
        
        return interaction_matrix
    
    def exercise_5_sequence_similarity_matrix(self):
        """
        Exercise 5: Sequence Similarity Matrix
        
        Task: Create and analyze sequence similarity matrices
        """
        print("\n=== Exercise 5: Sequence Similarity Matrix ===")
        
        # Create synthetic DNA sequences
        n_sequences = 25
        sequence_length = 50
        nucleotides = ['A', 'C', 'G', 'T']
        
        sequences = []
        for i in range(n_sequences):
            seq = ''.join(np.random.choice(nucleotides, size=sequence_length))
            sequences.append(seq)
        
        # TODO: Create similarity matrix
        # 1. Calculate pairwise similarity between all sequences
        # 2. Use different similarity measures (Hamming, Jaccard, etc.)
        # 3. Apply PCA to the similarity matrix
        # 4. Visualize sequence relationships
        
        print("TODO: Create sequence similarity matrix")
        print("1. Calculate pairwise similarities")
        print("2. Apply PCA to similarity matrix")
        print("3. Visualize sequence relationships")
        print("4. Identify sequence clusters")
        
        return sequences
    
    def exercise_6_eigenvalue_analysis(self):
        """
        Exercise 6: Eigenvalue Analysis
        
        Task: Analyze eigenvalues and eigenvectors of biological matrices
        """
        print("\n=== Exercise 6: Eigenvalue Analysis ===")
        
        # Create correlation matrix
        n_genes = 100
        correlation_matrix = np.random.random((n_genes, n_genes))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1)  # Diagonal = 1
        
        # TODO: Analyze eigenvalues
        # 1. Calculate eigenvalues and eigenvectors
        # 2. Plot eigenvalue spectrum
        # 3. Identify number of significant components
        # 4. Analyze eigenvector structure
        
        print("TODO: Analyze eigenvalues and eigenvectors")
        print("1. Calculate eigenvalues and eigenvectors")
        print("2. Plot eigenvalue spectrum")
        print("3. Determine number of significant components")
        print("4. Analyze eigenvector patterns")
        
        return correlation_matrix
    
    def run_all_exercises(self):
        """Run all exercises."""
        print("Running Linear Algebra Exercises for Bioinformatics\n")
        
        # Run exercises
        self.exercise_1_svd_implementation()
        self.exercise_2_pca_implementation()
        expression_data, tumor_genes = self.exercise_3_gene_expression_analysis()
        interaction_matrix = self.exercise_4_protein_network_analysis()
        sequences = self.exercise_5_sequence_similarity_matrix()
        correlation_matrix = self.exercise_6_eigenvalue_analysis()
        
        print("\n=== All Exercises Complete ===")
        print("Next steps:")
        print("1. Implement each exercise")
        print("2. Test your implementations")
        print("3. Compare with sklearn implementations")
        print("4. Interpret results in biological context")
        
        return {
            'expression_data': expression_data,
            'tumor_genes': tumor_genes,
            'interaction_matrix': interaction_matrix,
            'sequences': sequences,
            'correlation_matrix': correlation_matrix
        }

def main():
    """Main function to run exercises."""
    exercises = LinearAlgebraExercises()
    results = exercises.run_all_exercises()
    
    print("\nExercise data available in 'results' dictionary")
    print("Use this data to implement the exercises")

if __name__ == "__main__":
    main() 