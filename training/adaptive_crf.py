"""
Adaptive CRF Layer for Large Label Sets
Structured CRF with group-aware transition constraints for stable 161-label training.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple


class AdaptiveCRF(nn.Module):
    """
    Conditional Random Field with structured transitions for large label sets.
    
    Key Features:
    1. Group-aware transitions (entities in same group have higher transition probability)
    2. BIO constraint enforcement (I-X can only follow B-X or I-X)
    3. Structured initialization (prevent impossible transitions)
    4. Efficient forward/backward algorithm for 161 labels
    
    Used for sequence labeling with hierarchical label structure.
    """
    
    def __init__(
        self,
        num_labels: int = 161,
        entity_group_mapping: Optional[Dict[int, List[int]]] = None,
        pad_idx: int = -100
    ):
        """
        Args:
            num_labels: Total number of labels (including O)
            entity_group_mapping: Dict mapping group_id -> list of label_ids
            pad_idx: Padding index to ignore in loss computation
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.pad_idx = pad_idx
        self.entity_group_mapping = entity_group_mapping or {}
        
        # Transition parameters: [from_label, to_label]
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))
        
        # Initialize with structural constraints
        self._initialize_transitions()
    
    def _initialize_transitions(self):
        """
        Initialize transition matrix with structural constraints.
        
        Rules:
        1. O -> O (allowed)
        2. O -> B-X (allowed for any X)
        3. O -> I-X (disallowed - must start with B-X)
        4. B-X -> I-X (encouraged - continue entity)
        5. B-X -> B-Y (allowed - start new entity)
        6. B-X -> O (allowed - end entity)
        7. I-X -> I-X (encouraged - continue entity)
        8. I-X -> B-Y (allowed - start new entity)
        9. I-X -> I-Y where X != Y (disallowed - invalid BIO)
        10. Transitions within same group (higher probability)
        """
        with torch.no_grad():
            # Start with very negative values (discourage all)
            self.transitions.fill_(-10.0)
            
            # O (index 0) transitions
            self.transitions[0, 0] = 0.0  # O -> O (neutral)
            
            # O -> any B-X (allowed)
            for i in range(1, self.num_labels, 2):  # Odd indices = B-
                self.transitions[0, i] = 0.0
            
            # O -> any I-X (disallowed)
            for i in range(2, self.num_labels, 2):  # Even indices = I-
                self.transitions[0, i] = -10.0
            
            # For each entity (B-X, I-X pairs)
            for i in range(1, self.num_labels - 1, 2):
                b_idx = i      # B-X index
                i_idx = i + 1  # I-X index
                
                # B-X -> I-X (strongly encouraged)
                self.transitions[b_idx, i_idx] = 2.0
                
                # I-X -> I-X (strongly encouraged)
                self.transitions[i_idx, i_idx] = 2.0
                
                # B-X -> O (allowed)
                self.transitions[b_idx, 0] = 0.0
                
                # I-X -> O (allowed)
                self.transitions[i_idx, 0] = 0.0
                
                # B-X -> any other B-Y (slightly discouraged but allowed)
                for j in range(1, self.num_labels, 2):
                    if j != b_idx:
                        self.transitions[b_idx, j] = -1.0
                
                # I-X -> any B-Y (slightly discouraged but allowed)
                for j in range(1, self.num_labels, 2):
                    self.transitions[i_idx, j] = -1.0
                
                # I-X -> any I-Y where Y != X (strongly disallowed)
                for j in range(2, self.num_labels, 2):
                    if j != i_idx:
                        self.transitions[i_idx, j] = -20.0
            
            # Group-aware transitions (if mapping provided)
            if self.entity_group_mapping:
                for group_id, label_ids in self.entity_group_mapping.items():
                    # Encourage transitions within same group
                    for from_label in label_ids:
                        for to_label in label_ids:
                            if from_label != to_label:
                                # Check if valid BIO transition
                                from_is_b = from_label % 2 == 1
                                to_is_b = to_label % 2 == 1
                                
                                # B-X -> B-Y in same group (slight encouragement)
                                if from_is_b and to_is_b:
                                    self.transitions[from_label, to_label] += 0.5
                                # I-X -> B-Y in same group (slight encouragement)
                                elif not from_is_b and to_is_b:
                                    self.transitions[from_label, to_label] += 0.5
            
            # Start transitions
            # Encourage starting with O or B-X
            self.start_transitions[0] = 0.0  # O
            for i in range(1, self.num_labels, 2):  # B-X
                self.start_transitions[i] = 0.0
            for i in range(2, self.num_labels, 2):  # I-X (discourage)
                self.start_transitions[i] = -10.0
            
            # End transitions
            # Encourage ending with O or completing an entity
            self.end_transitions[0] = 0.0  # O
            for i in range(1, self.num_labels):
                self.end_transitions[i] = -0.5  # Slight penalty for incomplete
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Args:
            emissions: [batch, seq_len, num_labels] - Unary potentials
            tags: [batch, seq_len] - Ground truth labels
            mask: [batch, seq_len] - 1 for valid tokens, 0 for padding
            reduction: "mean", "sum", or "none"
        
        Returns:
            Negative log-likelihood loss
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        else:
            mask = mask.bool()
        
        # Compute log partition function (normalization)
        log_z = self._compute_log_partition(emissions, mask)
        
        # Compute score of ground truth sequence
        gold_score = self._compute_score(emissions, tags, mask)
        
        # Negative log-likelihood
        nll = log_z - gold_score
        
        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll
    
    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute score of a given tag sequence.
        
        Score = sum of emission scores + sum of transition scores
        """
        batch_size, seq_len = tags.shape
        device = emissions.device
        
        # Initialize scores
        scores = torch.zeros(batch_size, device=device)
        
        # Add start transition scores
        first_tags = tags[:, 0]
        scores += self.start_transitions[first_tags]
        
        # Add emission scores and transition scores
        for t in range(seq_len):
            # Get current tags
            current_tags = tags[:, t]
            
            # Add emission scores
            emit_scores = emissions[:, t].gather(1, current_tags.unsqueeze(1)).squeeze(1)
            scores += emit_scores * mask[:, t].float()
            
            # Add transition scores (if not last position)
            if t < seq_len - 1:
                next_tags = tags[:, t + 1]
                trans_scores = self.transitions[current_tags, next_tags]
                scores += trans_scores * mask[:, t + 1].float()
        
        # Add end transition scores (at last valid position)
        seq_lengths = mask.sum(dim=1)
        last_tag_indices = seq_lengths - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        scores += self.end_transitions[last_tags]
        
        return scores
    
    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log partition function using forward algorithm.
        
        Z = sum over all possible tag sequences of exp(score(sequence))
        log Z computed efficiently via dynamic programming
        """
        batch_size, seq_len, num_labels = emissions.shape
        
        # Initialize forward variables with start transitions + first emissions
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        # [batch, num_labels]
        
        # Forward pass
        for t in range(1, seq_len):
            # Broadcast for all transitions
            # alpha: [batch, num_labels, 1]
            # transitions: [num_labels, num_labels]
            # emissions: [batch, 1, num_labels]
            
            alpha_broadcast = alpha.unsqueeze(2)  # [batch, from_label, 1]
            emit_broadcast = emissions[:, t].unsqueeze(1)  # [batch, 1, to_label]
            trans_broadcast = self.transitions.unsqueeze(0)  # [1, from_label, to_label]
            
            # Compute all paths: alpha[t-1, from] + trans[from, to] + emit[t, to]
            scores = alpha_broadcast + trans_broadcast + emit_broadcast
            # [batch, from_label, to_label]
            
            # Log-sum-exp over from_label dimension
            alpha_new = torch.logsumexp(scores, dim=1)  # [batch, to_label]
            
            # Update alpha with masking
            alpha = torch.where(
                mask[:, t].unsqueeze(1),
                alpha_new,
                alpha
            )
        
        # Add end transitions
        alpha = alpha + self.end_transitions.unsqueeze(0)
        
        # Log partition function
        log_z = torch.logsumexp(alpha, dim=1)  # [batch]
        
        return log_z
    
    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Viterbi decoding to find most likely tag sequence.
        
        Args:
            emissions: [batch, seq_len, num_labels]
            mask: [batch, seq_len]
        
        Returns:
            List of predicted tag sequences (one per batch)
        """
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2],
                dtype=torch.bool,
                device=emissions.device
            )
        else:
            mask = mask.bool()
        
        batch_size, seq_len, num_labels = emissions.shape
        device = emissions.device
        
        # Initialize with start transitions + first emissions
        viterbi = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        # [batch, num_labels]
        
        # Backpointers for path reconstruction
        backpointers = []
        
        # Forward pass (Viterbi)
        for t in range(1, seq_len):
            # Broadcast for transitions
            viterbi_broadcast = viterbi.unsqueeze(2)  # [batch, from_label, 1]
            trans_broadcast = self.transitions.unsqueeze(0)  # [1, from_label, to_label]
            
            # Compute scores for all transitions
            scores = viterbi_broadcast + trans_broadcast  # [batch, from_label, to_label]
            
            # Find best previous label for each current label
            viterbi_new, backpointer = scores.max(dim=1)  # [batch, to_label]
            
            # Add emissions
            viterbi_new = viterbi_new + emissions[:, t]
            
            # Update with masking
            viterbi = torch.where(
                mask[:, t].unsqueeze(1),
                viterbi_new,
                viterbi
            )
            
            backpointers.append(backpointer)
        
        # Add end transitions
        viterbi = viterbi + self.end_transitions.unsqueeze(0)
        
        # Find best final label
        best_last_tags = viterbi.argmax(dim=1)  # [batch]
        
        # Backward pass to reconstruct paths
        best_paths = []
        
        for b in range(batch_size):
            # Get sequence length for this sample
            seq_len_b = mask[b].sum().item()
            
            # Start from best last tag
            path = [best_last_tags[b].item()]
            
            # Backtrack
            for t in range(len(backpointers) - 1, -1, -1):
                if t >= seq_len_b - 1:
                    continue
                prev_tag = backpointers[t][b, path[0]].item()
                path.insert(0, prev_tag)
            
            # Pad to full sequence length with O (index 0)
            while len(path) < seq_len:
                path.append(0)
            
            best_paths.append(path[:seq_len])
        
        return best_paths
    
    def get_transition_matrix(self) -> torch.Tensor:
        """Get current transition matrix (for visualization/debugging)"""
        return self.transitions.detach()
    
    def get_group_transition_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Analyze transition statistics per group.
        Useful for debugging and ensuring group-aware transitions work.
        """
        if not self.entity_group_mapping:
            return {}
        
        stats = {}
        trans_matrix = self.transitions.detach()
        
        for group_id, label_ids in self.entity_group_mapping.items():
            # Intra-group transitions (within same group)
            intra_transitions = []
            for from_label in label_ids:
                for to_label in label_ids:
                    if from_label != to_label:
                        intra_transitions.append(
                            trans_matrix[from_label, to_label].item()
                        )
            
            # Inter-group transitions (to other groups)
            inter_transitions = []
            all_other_labels = [
                l for gid, labels in self.entity_group_mapping.items()
                if gid != group_id
                for l in labels
            ]
            for from_label in label_ids:
                for to_label in all_other_labels:
                    inter_transitions.append(
                        trans_matrix[from_label, to_label].item()
                    )
            
            stats[group_id] = {
                'avg_intra': sum(intra_transitions) / len(intra_transitions) if intra_transitions else 0,
                'avg_inter': sum(inter_transitions) / len(inter_transitions) if inter_transitions else 0,
                'intra_minus_inter': (
                    (sum(intra_transitions) / len(intra_transitions)) -
                    (sum(inter_transitions) / len(inter_transitions))
                ) if intra_transitions and inter_transitions else 0
            }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Mock group mapping (from hierarchical classifier)
    group_mapping = {
        0: list(range(1, 15)),     # document_metadata
        1: list(range(15, 25)),    # supplier_info
        2: list(range(25, 33)),    # buyer_info
        3: list(range(33, 59)),    # financial_totals
        # ... etc
    }
    
    # Create CRF
    crf = AdaptiveCRF(
        num_labels=161,
        entity_group_mapping=group_mapping
    )
    
    # Mock data
    batch_size, seq_len = 4, 32
    emissions = torch.randn(batch_size, seq_len, 161)
    tags = torch.randint(0, 161, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Compute loss
    loss = crf(emissions, tags, mask)
    print(f"CRF Loss: {loss.item():.4f}")
    
    # Decode (inference)
    best_paths = crf.decode(emissions, mask)
    print(f"\nDecoded paths (first sample): {best_paths[0][:10]}...")
    
    # Analyze group transitions
    stats = crf.get_group_transition_stats()
    print("\nGroup transition statistics:")
    for group_id, group_stats in stats.items():
        print(f"  Group {group_id}:")
        print(f"    Avg intra-group: {group_stats['avg_intra']:.3f}")
        print(f"    Avg inter-group: {group_stats['avg_inter']:.3f}")
        print(f"    Difference: {group_stats['intra_minus_inter']:.3f}")
