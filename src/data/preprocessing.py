"""
Data preprocessing utilities for bioinformatics ML/AI applications.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class BiologicalDataPreprocessor:
    """
    Preprocessor for biological data with specialized methods for bioinformatics.
    """
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler to use ("standard", "minmax", or None)
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.label_encoder = None
        
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Transformed features and targets
        """
        # Handle scaling
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        # Handle label encoding for categorical targets
        if y is not None and self._is_categorical(y):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            
        return X, y
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame],
                 y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Transformed features and targets
        """
        if self.scaler is not None:
            X = self.scaler.transform(X)
            
        if y is not None and self.label_encoder is not None:
            y = self.label_encoder.transform(y)
            
        return X, y
    
    def _is_categorical(self, y: Union[np.ndarray, pd.Series]) -> bool:
        """Check if target variable is categorical."""
        return y.dtype == 'object' or len(np.unique(y)) < len(y) * 0.1


class SequencePreprocessor:
    """
    Preprocessor for biological sequences (DNA, RNA, proteins).
    """
    
    def __init__(self, sequence_type: str = "protein"):
        """
        Initialize sequence preprocessor.
        
        Args:
            sequence_type: Type of sequence ("dna", "rna", "protein")
        """
        self.sequence_type = sequence_type
        self._setup_encoding()
    
    def _setup_encoding(self):
        """Setup encoding dictionaries for different sequence types."""
        if self.sequence_type == "dna":
            self.encoding = {
                'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
                'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
                'N': [0.25, 0.25, 0.25, 0.25]  # Ambiguous
            }
        elif self.sequence_type == "rna":
            self.encoding = {
                'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
                'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1],
                'N': [0.25, 0.25, 0.25, 0.25]  # Ambiguous
            }
        elif self.sequence_type == "protein":
            # Standard amino acid encoding (simplified)
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            self.encoding = {aa: [1 if i == j else 0 for j in range(20)] 
                           for i, aa in enumerate(amino_acids)}
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode a biological sequence into numerical representation.
        
        Args:
            sequence: Input sequence string
            
        Returns:
            Encoded sequence as numpy array
        """
        encoded = []
        for char in sequence.upper():
            if char in self.encoding:
                encoded.append(self.encoding[char])
            else:
                # Handle unknown characters
                encoded.append([0] * len(next(iter(self.encoding.values()))))
        
        return np.array(encoded)
    
    def encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        Encode multiple sequences.
        
        Args:
            sequences: List of sequence strings
            
        Returns:
            Array of encoded sequences
        """
        return np.array([self.encode_sequence(seq) for seq in sequences])


def split_data(X: Union[np.ndarray, pd.DataFrame], 
               y: Union[np.ndarray, pd.Series],
               test_size: float = 0.2,
               val_size: float = 0.2,
               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                               np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Input features
        y: Target variable
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test 