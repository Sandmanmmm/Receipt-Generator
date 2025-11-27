"""
Randomizers - Utility functions for data randomization
"""
import random
from typing import List, Any, Optional, TypeVar
from datetime import datetime, timedelta

T = TypeVar('T')


class Randomizer:
    """Utility class for common randomization patterns"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize randomizer
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        self.seed = seed
    
    def choice(self, items: List[T]) -> T:
        """Randomly select one item from list"""
        return random.choice(items)
    
    def choices(self, items: List[T], k: int) -> List[T]:
        """Randomly select k items with replacement"""
        return random.choices(items, k=k)
    
    def sample(self, items: List[T], k: int) -> List[T]:
        """Randomly select k unique items without replacement"""
        return random.sample(items, k=k)
    
    def weighted_choice(self, items: List[T], weights: List[float]) -> T:
        """Select item based on weights"""
        return random.choices(items, weights=weights, k=1)[0]
    
    def random_int(self, min_val: int, max_val: int) -> int:
        """Generate random integer in range [min_val, max_val]"""
        return random.randint(min_val, max_val)
    
    def random_float(self, min_val: float, max_val: float, decimals: int = 2) -> float:
        """Generate random float in range [min_val, max_val]"""
        value = random.uniform(min_val, max_val)
        return round(value, decimals)
    
    def random_bool(self, probability: float = 0.5) -> bool:
        """Generate random boolean with specified probability of True"""
        return random.random() < probability
    
    def random_date(self, start_date: datetime, end_date: datetime) -> datetime:
        """Generate random date between start and end"""
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        return start_date + timedelta(days=random_days)
    
    def random_date_range(self, days_back: int = 90) -> datetime:
        """Generate random date going back specified days from today"""
        today = datetime.now()
        start_date = today - timedelta(days=days_back)
        return self.random_date(start_date, today)
    
    def shuffle(self, items: List[T]) -> List[T]:
        """Shuffle list in-place and return it"""
        random.shuffle(items)
        return items


class InvoiceRandomizer(Randomizer):
    """Specialized randomizer for invoice-specific values"""
    
    # Currency configurations
    CURRENCIES = {
        'USD': {'symbol': '$', 'code': 'USD'},
        'EUR': {'symbol': '€', 'code': 'EUR'},
        'GBP': {'symbol': '£', 'code': 'GBP'},
        'JPY': {'symbol': '¥', 'code': 'JPY'},
        'CAD': {'symbol': 'C$', 'code': 'CAD'},
        'AUD': {'symbol': 'A$', 'code': 'AUD'},
        'CHF': {'symbol': 'CHF', 'code': 'CHF'},
        'CNY': {'symbol': '¥', 'code': 'CNY'},
    }
    
    # Common tax rates by region
    TAX_RATES = {
        'US': [0, 5, 7.5, 10],
        'EU': [19, 20, 21, 23],
        'UK': [0, 5, 20],
        'CA': [0, 5, 12, 13, 15],
        'AU': [0, 10],
    }
    
    # Common payment terms
    PAYMENT_TERMS = [
        'Net 7', 'Net 10', 'Net 15', 'Net 30', 'Net 45', 'Net 60',
        'Due on Receipt', '2/10 Net 30', '1/10 Net 30'
    ]
    
    # Discount percentages
    DISCOUNT_RATES = [5, 10, 15, 20, 25]
    
    def random_currency(self, currencies: Optional[List[str]] = None) -> dict:
        """
        Get random currency configuration
        
        Args:
            currencies: List of currency codes (default: all)
            
        Returns:
            Dictionary with 'symbol' and 'code'
        """
        if currencies is None:
            currencies = list(self.CURRENCIES.keys())
        
        currency_code = self.choice(currencies)
        return self.CURRENCIES[currency_code]
    
    def random_tax_rate(self, region: str = 'US') -> float:
        """
        Get random tax rate for region
        
        Args:
            region: Region code ('US', 'EU', 'UK', 'CA', 'AU')
            
        Returns:
            Tax rate as float
        """
        rates = self.TAX_RATES.get(region, [0, 5, 10, 15, 20])
        return self.choice(rates)
    
    def random_payment_term(self) -> str:
        """Get random payment term"""
        return self.choice(self.PAYMENT_TERMS)
    
    def random_discount_rate(self) -> float:
        """Get random discount rate"""
        return self.choice(self.DISCOUNT_RATES)
    
    def random_quantity(self, min_qty: int = 1, max_qty: int = 20) -> int:
        """Generate random quantity for line items"""
        # Weight towards smaller quantities
        weights = [1.0 / (i + 1) for i in range(max_qty - min_qty + 1)]
        quantities = list(range(min_qty, max_qty + 1))
        return self.weighted_choice(quantities, weights)
    
    def random_price(self, min_price: float = 10.0, max_price: float = 1000.0,
                    price_points: bool = True) -> float:
        """
        Generate random price
        
        Args:
            min_price: Minimum price
            max_price: Maximum price
            price_points: If True, round to common price points (.00, .49, .95, .99)
            
        Returns:
            Random price
        """
        price = self.random_float(min_price, max_price, decimals=2)
        
        if price_points and self.random_bool(0.7):
            # Round to common price points
            integer_part = int(price)
            decimal_part = self.choice([0.00, 0.49, 0.95, 0.99])
            price = integer_part + decimal_part
        
        return price
    
    def random_invoice_number(self, prefix: str = 'INV', year: bool = True,
                            digits: int = 4) -> str:
        """
        Generate random invoice number
        
        Args:
            prefix: Invoice number prefix
            year: Include year
            digits: Number of digits
            
        Returns:
            Invoice number string
        """
        parts = [prefix]
        
        if year:
            current_year = datetime.now().year
            parts.append(str(current_year))
        
        number = str(self.random_int(1, 10**digits - 1)).zfill(digits)
        parts.append(number)
        
        return '-'.join(parts)
    
    def random_due_date(self, invoice_date: datetime, 
                       term_days: Optional[int] = None) -> datetime:
        """
        Generate due date based on invoice date
        
        Args:
            invoice_date: Invoice date
            term_days: Payment term in days (random if None)
            
        Returns:
            Due date
        """
        if term_days is None:
            term_days = self.choice([7, 14, 30, 45, 60])
        
        return invoice_date + timedelta(days=term_days)


__all__ = ['Randomizer', 'InvoiceRandomizer']
