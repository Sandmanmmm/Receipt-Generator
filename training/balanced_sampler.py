"""
Domain-Balanced Sampler for LayoutLMv3 Training
Ensures rare entities appear frequently enough for learning.
"""

import random
from typing import List, Dict
from torch.utils.data import Sampler


class DomainBalancedSampler(Sampler):
    """
    Stratified sampler that balances across:
    1. Invoice domains (General, SaaS, Telecom, Logistics, etc.)
    2. Entity rareness (frequent vs rare entities)
    
    Ensures each batch contains:
    - Mix of all domains
    - At least N examples with rare entities (CAGE_CODE, LOT_NUMBER, etc.)
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 16,
        domain_key: str = "domain",  # Key in dataset samples
        rare_entity_key: str = "has_rare_entities",
        min_rare_entities_per_batch: int = 4,
        shuffle: bool = True
    ):
        """
        Args:
            dataset: Dataset with domain and rare entity annotations
            batch_size: Batch size for training
            domain_key: Key in dataset dict for domain ("General", "SaaS", etc.)
            rare_entity_key: Key for bool indicating rare entities present
            min_rare_entities_per_batch: Minimum samples with rare entities per batch
            shuffle: Whether to shuffle within domains
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.domain_key = domain_key
        self.rare_entity_key = rare_entity_key
        self.min_rare = min_rare_entities_per_batch
        self.shuffle = shuffle
        
        # Build domain indices
        self.domain_indices = self._build_domain_indices()
        
        # Build rare entity indices
        self.rare_indices = self._build_rare_indices()
        
    def _build_domain_indices(self) -> Dict[str, List[int]]:
        """Group sample indices by domain"""
        domain_indices = {}
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            domain = sample.get(self.domain_key, "General")
            
            if domain not in domain_indices:
                domain_indices[domain] = []
            domain_indices[domain].append(idx)
        
        return domain_indices
    
    def _build_rare_indices(self) -> List[int]:
        """Get indices of samples containing rare entities"""
        rare_indices = []
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            if sample.get(self.rare_entity_key, False):
                rare_indices.append(idx)
        
        return rare_indices
    
    def __iter__(self):
        """Generate batches with domain and rare entity balance"""
        
        # Shuffle within each domain
        if self.shuffle:
            for indices in self.domain_indices.values():
                random.shuffle(indices)
            random.shuffle(self.rare_indices)
        
        # Create domain iterators
        domain_iters = {
            domain: iter(indices)
            for domain, indices in self.domain_indices.items()
        }
        rare_iter = iter(self.rare_indices)
        
        batch = []
        domains_cycle = list(self.domain_indices.keys())
        domain_idx = 0
        
        while True:
            # Add rare entity samples first (up to min_rare)
            rare_count = 0
            while rare_count < self.min_rare and len(batch) < self.batch_size:
                try:
                    idx = next(rare_iter)
                    if idx not in batch:  # Avoid duplicates
                        batch.append(idx)
                        rare_count += 1
                except StopIteration:
                    rare_iter = iter(self.rare_indices)
                    if self.shuffle:
                        random.shuffle(self.rare_indices)
                    break
            
            # Fill rest of batch with round-robin from domains
            while len(batch) < self.batch_size:
                domain = domains_cycle[domain_idx % len(domains_cycle)]
                domain_idx += 1
                
                try:
                    idx = next(domain_iters[domain])
                    if idx not in batch:  # Avoid duplicates
                        batch.append(idx)
                except StopIteration:
                    # Reset domain iterator
                    if self.shuffle:
                        random.shuffle(self.domain_indices[domain])
                    domain_iters[domain] = iter(self.domain_indices[domain])
            
            # Yield batch
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            
            # Check if all domains exhausted
            all_exhausted = all(
                len(indices) == 0
                for indices in self.domain_indices.values()
            )
            if all_exhausted and len(batch) == 0:
                break
        
        # Yield remaining samples
        if len(batch) > 0:
            yield batch
    
    def __len__(self):
        """Total number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# Example usage
if __name__ == "__main__":
    # Mock dataset
    class MockDataset:
        def __init__(self):
            self.data = [
                {"domain": "General", "has_rare_entities": False},
                {"domain": "SaaS", "has_rare_entities": True},
                {"domain": "Telecom", "has_rare_entities": False},
                {"domain": "Logistics", "has_rare_entities": True},
                # ... 250K samples
            ] * 1000
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = MockDataset()
    sampler = DomainBalancedSampler(
        dataset,
        batch_size=16,
        min_rare_entities_per_batch=4
    )
    
    # Test sampling
    for batch_idx, batch_indices in enumerate(sampler):
        print(f"Batch {batch_idx}: {len(batch_indices)} samples")
        
        # Check domain distribution
        domains = [dataset[i]["domain"] for i in batch_indices]
        rare_count = sum(dataset[i]["has_rare_entities"] for i in batch_indices)
        
        print(f"  Domains: {set(domains)}")
        print(f"  Rare entities: {rare_count}/{len(batch_indices)}")
        
        if batch_idx >= 5:
            break
