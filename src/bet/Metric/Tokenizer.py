import multiprocessing
from functools import partial
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed


class DepthFirstInteractionTokenizer():
    def __init__(self, 
        max_vocab_size=1000, 
        interaction_degree=5, 
        min_frequency=5, 
        max_candidates_to_evaluate=10000, 
        beam_width=100,
        min_section_sizes=None,
        n_jobs=-1
    ):
        self.max_vocab_size = max_vocab_size
        self.interaction_degree = interaction_degree
        self.min_frequency = min_frequency
        self.max_candidates_to_evaluate = max_candidates_to_evaluate
        self.beam_width = beam_width  # Number of candidates to maintain in beam search
        self.min_section_sizes = min_section_sizes  # Dict of {degree: min_size} for each interaction degree
        self.n_jobs = n_jobs  # Number of parallel jobs (-1 for all cores)
        self.feature_vocab = {}
        self.inverse_vocab = {}
        self.feature_frequencies = None
        
        # For faster transform
        self._feature_indices = {}
        self._interaction_indices = {}
        self.feature_names = None  # Will store feature names mapping

        # Section masks
        self.section_masks = {}  # Dictionary of masks for each degree
        self.section_ranges = {}  # Dictionary of index ranges for each degree
        
    def _count_base_features(self, 
        X_list, 
        feature_names=None
    ):
        """Count frequencies of base features using a sparse matrix approach"""
        # Determine max feature index
        max_feature_idx = 0
        for sample in X_list:
            max_feature_idx = max(max_feature_idx, len(sample))
        
        # Create binary sparse matrix to represent feature presence
        n_samples = len(X_list)
        rows = []
        cols = []
        data = []
        
        # Track active features directly
        active_features = set()
        
        for sample_idx, sample in enumerate(X_list):
            # Get indices of non-zero elements
            for feat_idx, val in enumerate(sample):
                if val != 0:
                    rows.append(sample_idx)
                    cols.append(feat_idx)
                    data.append(1)
                    active_features.add(feat_idx)
        
        # Create sparse matrix
        feature_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_samples, max_feature_idx))
        
        # Sum across samples to get feature frequencies
        feature_counts = {}
        feature_sums = np.array(feature_matrix.sum(axis=0)).flatten()
        
        # Use feature names if provided
        if feature_names:
            # Validate feature names length
            if len(feature_names) < max_feature_idx:
                print(f"Warning: Only {len(feature_names)} names provided for {max_feature_idx} features")
                # Extend feature_names with default names if needed
                feature_names = list(feature_names)  # Ensure it's a list
                feature_names.extend([f"feature_{i}" for i in range(len(feature_names), max_feature_idx)])
            
            self.feature_names = feature_names
            for i, count in enumerate(feature_sums):
                if i < len(feature_names) and count > 0:
                    feature_counts[feature_names[i]] = int(count)
        else:
            # Use default feature naming (f_0, f_1, etc.)
            self.feature_names = [f"f_{i}" for i in range(max_feature_idx)]
            for i, count in enumerate(feature_sums):
                if count > 0:
                    feature_counts[f"f_{i}"] = int(count)
        
        # Verify all active features are counted
        if len(active_features) != len(feature_counts):
            # Ensure all active features are in the dictionary
            for idx in active_features:
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    if feature_name not in feature_counts:
                        feature_counts[feature_name] = 1  # Assign minimum count
        
        return feature_counts, feature_matrix
    
    def _filter_high_target_samples(self,
        y, 
        feature_matrix
    ):
        """Filter samples with target ≥ 3 and return their indices and a submatrix"""        
        high_target_mask = y >= 3
        high_target_indices = np.where(high_target_mask)[0]
        high_target_matrix = feature_matrix[high_target_indices]
        
        return high_target_indices, high_target_matrix
    
    def _evaluate_interaction_support(self, 
        feature_indices, 
        feature_matrix
    ):
        """Evaluate support (frequency) of a specific interaction"""
        # Get sets of samples containing each feature
        sample_sets = []
        for feat_idx in feature_indices:
            if feat_idx < feature_matrix.shape[1]:
                # Get indices of samples containing this feature
                samples_with_feature = set(feature_matrix[:, feat_idx].nonzero()[0])
                sample_sets.append(samples_with_feature)
            else:
                return 0  # Feature index out of bounds
                
        # Return intersection size (support)
        if not sample_sets:
            return 0
        return len(set.intersection(*sample_sets))
    
    def _find_high_degree_interactions_depth_first(self, 
        feature_matrix, 
        base_feature_counts, 
        target_degree
    ):
        """Find interactions using depth-first beam search to directly target high-degree interactions"""
        # Map feature names to indices for matrix operations
        name_to_idx = {}
        for feature_name in base_feature_counts.keys():
            # For default names (f_0, f_1...), extract the number
            if feature_name.startswith('f_') and feature_name[2:].isdigit():
                name_to_idx[feature_name] = int(feature_name[2:])
            # For custom feature names, find the index in feature_names
            else:
                try:
                    idx = self.feature_names.index(feature_name)
                    name_to_idx[feature_name] = idx
                except ValueError:
                    print(f"Warning: Feature {feature_name} not found in feature_names")
                    continue
        
        # Sort features by frequency for initial beam
        sorted_features = sorted(base_feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create initial beam with individual features (name, [indices], count)
        beam = []
        for feature_name, count in sorted_features[:self.beam_width]:
            if feature_name in name_to_idx:
                idx = name_to_idx[feature_name]
                beam.append((feature_name, [idx], count))
        
        # Precompute sample sets for top features to avoid recomputation
        feature_to_samples = {}
        for feature_name, _ in sorted_features[:min(1000, len(sorted_features))]:
            if feature_name in name_to_idx:
                feat_idx = name_to_idx[feature_name]
                if feat_idx < feature_matrix.shape[1]:
                    feature_to_samples[feat_idx] = set(feature_matrix[:, feat_idx].nonzero()[0])
        
        # Track the best interactions found
        best_interactions = []
        candidates_evaluated = 0
        
        # For each level from 2 to target_degree
        for current_degree in range(2, target_degree + 1):
            next_beam = []
            
            # Expand each candidate in the current beam
            for last_name, current_features, support in beam:
                # Get last added index to ensure sorted order
                last_idx = current_features[-1]
                
                # Consider adding features that appear after the last added one
                potential_features = [
                    (f, name_to_idx[f]) for f, _ in sorted_features 
                    if f in name_to_idx and name_to_idx[f] > last_idx
                ]
                
                # Limit the number of potential features to prevent explosion
                for next_name, next_idx in potential_features[:100]:
                    candidates_evaluated += 1
                    if candidates_evaluated > self.max_candidates_to_evaluate:
                        break
                        
                    # Get samples containing this new feature
                    if next_idx in feature_to_samples:
                        samples_with_next = feature_to_samples[next_idx]
                    else:
                        if next_idx < feature_matrix.shape[1]:
                            samples_with_next = set(feature_matrix[:, next_idx].nonzero()[0])
                            feature_to_samples[next_idx] = samples_with_next
                        else:
                            continue  # Skip features with index out of bounds
                    
                    # Calculate support of the expanded combination
                    # First get the intersection of samples for current features
                    if current_degree == 2:  # Just starting with 1 feature
                        current_samples = feature_to_samples[current_features[0]]
                    else:
                        # Use precomputed support
                        current_samples = set.intersection(*[feature_to_samples[f] for f in current_features])
                    
                    # Calculate new support with the additional feature
                    new_support = len(current_samples.intersection(samples_with_next))
                    
                    if new_support >= self.min_frequency:
                        # Create new candidate
                        new_features = current_features + [next_idx]
                        
                        # Add to next beam
                        next_beam.append((next_name, new_features, new_support))
                        
                        # If we've reached the target degree, add to best interactions
                        if current_degree == target_degree:
                            # Create interaction key using feature names
                            feature_names = [self.feature_names[idx] for idx in new_features]
                            interaction_key = " x ".join(feature_names)
                            best_interactions.append((interaction_key, new_support))
            
            # If we're at the target degree, break the loop
            if current_degree == target_degree:
                break
                
            # Prepare beam for next iteration - take top candidates by support
            beam = sorted(next_beam, key=lambda x: x[2], reverse=True)[:self.beam_width]
            
            if not beam:
                break

        # Sort final interactions by support
        best_interactions.sort(key=lambda x: x[1], reverse=True)
        
        return best_interactions
                
    def _serialize_feature_matrix(self, 
        feature_matrix
    ):
        """Serialize sparse matrix for multiprocessing"""
        return {
            'data': feature_matrix.data,
            'indices': feature_matrix.indices,
            'indptr': feature_matrix.indptr,
            'shape': feature_matrix.shape
        }
    
    def _deserialize_feature_matrix(self, 
        serialized
    ):
        """Deserialize sparse matrix from multiprocessing"""
        return sp.csr_matrix(
            (serialized['data'], serialized['indices'], serialized['indptr']),
            shape=serialized['shape']
        )
        
    def _find_high_degree_interactions_parallel(self, 
        feature_matrix, 
        base_feature_counts, 
        n_jobs=-1
    ):
        """Parallelize interaction search across different degrees"""
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
            
        degrees = list(range(2, self.interaction_degree + 1))
        
        # For very large matrices, we might need to serialize/deserialize
        # but for most cases direct sharing should work fine with joblib
        
        # Create a partial function with common parameters
        search_func = partial(
            self._find_high_degree_interactions_depth_first,
            feature_matrix=feature_matrix,
            base_feature_counts=base_feature_counts
        )
        
        # Run in parallel
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(search_func)(degree) for degree in degrees
        )
        
        # Process results
        interactions_by_degree = {
            degree: result 
            for degree, result in zip(degrees, results)
        }
        all_interactions = []
        for interactions in results:
            all_interactions.extend(interactions)
            
        return all_interactions, interactions_by_degree
    
    def fit(self,
        X_list,
        y,
        feature_names=None
    ):
        """Build vocabulary with dedicated sections for each interaction degree"""
        if y is None:
            raise ValueError("Target values (y) must be provided for fitting")
            
        # Ensure y is a numpy array
        y = np.asarray(y)
        
        # Count base features and create sparse feature matrix
        base_feature_counts, feature_matrix = self._count_base_features(X_list, feature_names)
        self.feature_frequencies = base_feature_counts
        
        # Filter high target samples
        high_target_indices, high_target_matrix = self._filter_high_target_samples(y, feature_matrix)
        
        # Initialize section sizes
        if self.min_section_sizes is None:
            # Default to roughly equal allocation after base features
            self.min_section_sizes = {}
            base_size = len(base_feature_counts)
            remaining = self.max_vocab_size - base_size
            # Equal allocation for each degree from 2 to interaction_degree
            n_degrees = self.interaction_degree - 1  # From 2 to interaction_degree
            if n_degrees > 0:
                size_per_degree = remaining // n_degrees
                for degree in range(2, self.interaction_degree + 1):
                    self.min_section_sizes[degree] = size_per_degree
        
        # Map degrees to sections in the vocabulary
        idx = 0
        section_start_indices = {}
        
        # SECTION 1: Add all base features (degree 1)
        section_start_indices[1] = idx
        for feature in sorted(base_feature_counts.keys()):
            self.feature_vocab[feature] = idx
            self.inverse_vocab[idx] = feature
            
            # Store feature index mapping for faster transform
            if feature.startswith('f_') and feature[2:].isdigit():
                self._feature_indices[feature] = int(feature[2:])
            else:
                try:
                    self._feature_indices[feature] = self.feature_names.index(feature)
                except ValueError:
                    print(f"Warning: Feature {feature} not found in feature_names")
                    self._feature_indices[feature] = -1  # Invalid index
            
            idx += 1
        
        # Find all interactions for each degree if we have high-target samples
        all_interactions_by_degree = {}
        if self.interaction_degree >= 2 and len(high_target_indices) > 0:
            # Find interactions for each degree
            for degree in range(2, self.interaction_degree + 1):
                degree_interactions = self._find_high_degree_interactions_depth_first(
                    high_target_matrix, base_feature_counts, degree)
                all_interactions_by_degree[degree] = degree_interactions
        
        # Now create vocabulary sections for each interaction degree
        for degree in range(2, self.interaction_degree + 1):
            section_start_indices[degree] = idx
            
            # Get interactions for this degree
            degree_interactions = all_interactions_by_degree.get(degree, [])
            
            # Determine size for this section
            min_size = self.min_section_sizes.get(degree, 0)
            # Don't exceed available interactions
            section_size = min(min_size, len(degree_interactions))
            
            # Don't exceed total vocabulary size
            remaining_capacity = self.max_vocab_size - idx
            section_size = min(section_size, remaining_capacity)
            
            if section_size > 0:
                
                # Take top interactions by support
                top_interactions = degree_interactions[:section_size]
                
                # Add to vocabulary
                for feature, _ in top_interactions:
                    self.feature_vocab[feature] = idx
                    self.inverse_vocab[idx] = feature
                    
                    # Store interaction components for faster transform
                    components = feature.split(" x ")
                    component_indices = []
                    for comp in components:
                        if comp in self._feature_indices:
                            component_indices.append(self._feature_indices[comp])
                        elif comp.startswith('f_') and comp[2:].isdigit():
                            component_indices.append(int(comp[2:]))
                        else:
                            try:
                                component_indices.append(self.feature_names.index(comp))
                            except ValueError:
                                print(f"Warning: Component {comp} not found in feature_names")
                                component_indices.append(-1)  # Invalid index
                                
                    self._interaction_indices[idx] = component_indices
                    idx += 1
        
        # Mark section end indices
        section_end_indices = {}
        for degree in sorted(section_start_indices.keys()):
            next_degree = degree + 1
            if next_degree in section_start_indices:
                section_end_indices[degree] = section_start_indices[next_degree] - 1
            else:
                section_end_indices[degree] = idx - 1
        
        # Create masks and store section ranges
        self._create_section_masks(section_start_indices, section_end_indices)
        
        return self
        
    def _create_section_masks(self, 
        start_indices, 
        end_indices
    ):
        """Create binary masks for each interaction degree section"""
        n_features = len(self.feature_vocab)
        
        # Create masks and store ranges
        for degree in sorted(start_indices.keys()):
            start = start_indices[degree]
            end = end_indices[degree]
            
            # Create binary mask (1 for this degree, 0 elsewhere)
            mask = np.zeros(n_features, dtype=bool)
            mask[start:end+1] = True
            self.section_masks[degree] = mask
            
            # Store range for easy access
            self.section_ranges[degree] = (start, end)
            
    def get_section_mask(self, 
        degree
    ):
        """Get binary mask for a specific interaction degree"""
        if degree in self.section_masks:
            return self.section_masks[degree]
        else:
            raise ValueError(f"No section exists for degree {degree}")
    
    def get_section_indices(self, 
        degree
    ):
        """Get start and end indices for a specific interaction degree"""
        if degree in self.section_ranges:
            return self.section_ranges[degree]
        else:
            raise ValueError(f"No section exists for degree {degree}")
    
    def transform_with_sections(self, 
        X_list
    ):
        """Transform data and return both the encoded matrix and section masks"""
        encoded = self.transform(X_list)
        return encoded, self.section_masks
    
    def transform(self, X_list, use_masks=True, y=None):
        """Transform list of feature arrays to a CSR sparse matrix"""
        n_samples = len(X_list)
        n_features = len(self.feature_vocab)
        
        # Create a dense array first, then convert to sparse
        encoded_dense = np.zeros((n_samples, n_features), dtype=np.float32)
        
        # Pre-process feature keys to avoid string operations in the loop
        base_feature_indices = {}
        for feature, idx in self.feature_vocab.items():
            if ' x ' not in feature:  # Base feature
                if feature in self._feature_indices:
                    feat_idx = self._feature_indices[feature]
                    if feat_idx >= 0:  # Valid index
                        base_feature_indices[feat_idx] = idx
        
        # Process each sample
        for sample_idx, sample in enumerate(X_list):
            # Get non-zero features and values
            non_zero_indices = np.nonzero(sample)[0]
            
            # Skip empty samples
            if len(non_zero_indices) == 0:
                continue
                
            # Process base features directly
            for feat_idx in non_zero_indices:
                if feat_idx < len(sample) and feat_idx in base_feature_indices:
                    vocab_idx = base_feature_indices[feat_idx]
                    encoded_dense[sample_idx, vocab_idx] = sample[feat_idx]
            
            # Process interaction features - only if we have enough active features
            if len(non_zero_indices) >= 2:
                # Get the values for later use
                active_values = {idx: sample[idx] for idx in non_zero_indices}
                active_indices_set = set(non_zero_indices)
                
                # Check each interaction feature
                for feature_idx, components in self._interaction_indices.items():
                    # Check if all components are present in this sample
                    if all(comp in active_indices_set for comp in components):
                        # Calculate interaction value as product
                        interaction_value = np.prod([active_values[comp] for comp in components])
                        encoded_dense[sample_idx, feature_idx] = interaction_value
        
        if use_masks and hasattr(self, 'section_masks') and self.section_masks:
            encoded_dense = np.concatenate(
                [
                    encoded_dense * self.section_masks[i]
                    for i in self.section_masks.keys()
                    if len(range(*self.section_ranges[i])) > 0
                ]
            )
            if y is not None:
                y = np.concatenate(
                    [
                        y for i in self.section_masks.keys()
                        if len(range(*self.section_ranges[i])) > 0
                    ]
                )
        
        # Convert dense array to CSR format (much more efficient for sparse data)
        encoded_sparse = sp.csr_matrix(encoded_dense)
        
        return encoded_sparse, y
    
    def decode_features(self, 
        feature_indices
    ):
        """Convert encoded feature indices back to original feature names"""
        return [self.inverse_vocab.get(idx, f"unknown_{idx}") for idx in feature_indices]
    
    def get_feature_names(self):
        """Return all feature names in the vocabulary"""
        return list(self.inverse_vocab.values())
    
    def get_feature_importance(self, 
        indices: List[int]
    ) -> List[Tuple[str, float]]:
        """Return feature importance based on frequency for the given indices"""
        result = []
        for idx in indices:
            feature_name = self.inverse_vocab.get(idx)
            if feature_name:
                if " x " in feature_name:
                    # For interaction features, calculate average frequency of components
                    components = feature_name.split(" x ")
                    avg_freq = np.mean([self.feature_frequencies.get(comp, 0) for comp in components])
                    result.append((feature_name, avg_freq))
                else:
                    # For base features, use their frequency
                    result.append((feature_name, self.feature_frequencies.get(feature_name, 0)))
            else:
                result.append((f"unknown_{idx}", 0))
        
        return result