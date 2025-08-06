"""
Feature engineering utilities for bioinformatics ML/AI applications.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re


class BiologicalFeatureEngineer:
    """
    Feature engineer specialized for biological data.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
        
    def extract_sequence_features(self, sequences: List[str], 
                                sequence_type: str = "protein") -> pd.DataFrame:
        """
        Extract features from biological sequences.
        
        Args:
            sequences: List of sequence strings
            sequence_type: Type of sequence ("dna", "rna", "protein")
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        
        for seq in sequences:
            seq_features = {}
            
            # Basic sequence features
            seq_features['length'] = len(seq)
            seq_features['gc_content'] = self._calculate_gc_content(seq) if sequence_type in ["dna", "rna"] else None
            
            # Amino acid composition for proteins
            if sequence_type == "protein":
                aa_composition = self._calculate_aa_composition(seq)
                seq_features.update(aa_composition)
                
                # Physicochemical properties
                phys_props = self._calculate_physicochemical_properties(seq)
                seq_features.update(phys_props)
            
            features.append(seq_features)
        
        return pd.DataFrame(features)
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content for DNA/RNA sequences."""
        gc_count = sequence.upper().count('G') + sequence.upper().count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0
    
    def _calculate_aa_composition(self, sequence: str) -> Dict[str, float]:
        """Calculate amino acid composition."""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        composition = {}
        
        for aa in amino_acids:
            composition[f'aa_{aa}'] = sequence.upper().count(aa) / len(sequence)
        
        return composition
    
    def _calculate_physicochemical_properties(self, sequence: str) -> Dict[str, float]:
        """Calculate physicochemical properties of protein sequences."""
        # Simplified physicochemical properties
        properties = {
            'hydrophobicity': 0.0,
            'charge': 0.0,
            'polarity': 0.0
        }
        
        # Hydrophobicity scale (simplified)
        hydrophobic_aas = 'ACFILMPVWY'
        charged_aas = {'R': 1, 'K': 1, 'H': 0.1, 'D': -1, 'E': -1}
        polar_aas = 'DEHKNQRST'
        
        for aa in sequence.upper():
            if aa in hydrophobic_aas:
                properties['hydrophobicity'] += 1
            if aa in charged_aas:
                properties['charge'] += charged_aas[aa]
            if aa in polar_aas:
                properties['polarity'] += 1
        
        # Normalize by sequence length
        seq_len = len(sequence)
        if seq_len > 0:
            properties['hydrophobicity'] /= seq_len
            properties['polarity'] /= seq_len
        
        return properties


class ExpressionFeatureEngineer:
    """
    Feature engineer for gene expression data.
    """
    
    def __init__(self):
        """Initialize the expression feature engineer."""
        self.pca = None
        
    def extract_expression_features(self, expression_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from gene expression matrix.
        
        Args:
            expression_matrix: DataFrame with genes as columns and samples as rows
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=expression_matrix.index)
        
        # Statistical features
        features['mean_expression'] = expression_matrix.mean(axis=1)
        features['std_expression'] = expression_matrix.std(axis=1)
        features['max_expression'] = expression_matrix.max(axis=1)
        features['min_expression'] = expression_matrix.min(axis=1)
        features['expression_range'] = features['max_expression'] - features['min_expression']
        
        # Quantile features
        features['q25_expression'] = expression_matrix.quantile(0.25, axis=1)
        features['q75_expression'] = expression_matrix.quantile(0.75, axis=1)
        features['iqr_expression'] = features['q75_expression'] - features['q25_expression']
        
        # Zero expression features
        features['zero_genes'] = (expression_matrix == 0).sum(axis=1)
        features['zero_ratio'] = features['zero_genes'] / expression_matrix.shape[1]
        
        return features
    
    def apply_pca(self, expression_matrix: pd.DataFrame, 
                  n_components: int = 50) -> pd.DataFrame:
        """
        Apply PCA to gene expression data.
        
        Args:
            expression_matrix: Gene expression matrix
            n_components: Number of principal components
            
        Returns:
            DataFrame with principal components
        """
        self.pca = PCA(n_components=min(n_components, expression_matrix.shape[1]))
        pca_features = self.pca.fit_transform(expression_matrix)
        
        # Create column names
        pc_names = [f'PC_{i+1}' for i in range(pca_features.shape[1])]
        
        return pd.DataFrame(pca_features, 
                          index=expression_matrix.index, 
                          columns=pc_names)


class MolecularFeatureEngineer:
    """
    Feature engineer for molecular data (QSAR, drug discovery).
    """
    
    def __init__(self):
        """Initialize the molecular feature engineer."""
        pass
    
    def calculate_molecular_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Calculate molecular descriptors from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            DataFrame with molecular descriptors
        """
        # This is a simplified implementation
        # In practice, you would use RDKit or similar library
        descriptors = []
        
        for smiles in smiles_list:
            desc = self._calculate_simple_descriptors(smiles)
            descriptors.append(desc)
        
        return pd.DataFrame(descriptors)
    
    def _calculate_simple_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate simple molecular descriptors."""
        # Simplified molecular descriptors
        descriptors = {
            'molecular_weight': len(smiles) * 14.0,  # Rough approximation
            'atom_count': len(re.findall(r'[A-Z][a-z]?', smiles)),
            'ring_count': smiles.count('c') + smiles.count('C'),
            'heteroatom_count': len(re.findall(r'[NOSPF]', smiles)),
            'aromatic_rings': smiles.count('c'),
            'aliphatic_rings': smiles.count('C') - smiles.count('c')
        }
        
        return descriptors


def create_kmer_features(sequences: List[str], k: int = 3) -> pd.DataFrame:
    """
    Create k-mer features from sequences.
    
    Args:
        sequences: List of sequence strings
        k: Length of k-mers
        
    Returns:
        DataFrame with k-mer features
    """
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


def extract_evolutionary_features(sequences: List[str], 
                                alignment_scores: List[float]) -> pd.DataFrame:
    """
    Extract evolutionary features from sequences.
    
    Args:
        sequences: List of sequence strings
        alignment_scores: List of alignment scores
        
    Returns:
        DataFrame with evolutionary features
    """
    features = pd.DataFrame({
        'alignment_score': alignment_scores,
        'sequence_length': [len(seq) for seq in sequences],
        'conservation_score': [score / len(seq) for score, seq in zip(alignment_scores, sequences)]
    })
    
    return features 