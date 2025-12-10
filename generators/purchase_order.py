"""
Purchase Order Data Models

Defines data structures for B2B purchase order documents used for ordering
products from suppliers. Critical for Shopify product upload OCR training.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional, Dict
from enum import Enum


class POStatus(Enum):
    """Purchase order status options"""
    DRAFT = "Draft"
    SENT = "Sent"
    CONFIRMED = "Confirmed"
    IN_PRODUCTION = "In Production"
    SHIPPED = "Shipped"
    DELIVERED = "Delivered"
    RECEIVED = "Received"
    CANCELLED = "Cancelled"


class POType(Enum):
    """Purchase order template types"""
    ALIBABA = "alibaba"
    DROPSHIP = "dropship"
    DOMESTIC = "domestic_distributor"
    MANUFACTURER = "manufacturer_direct"
    FASHION = "fashion_wholesale"
    ELECTRONICS = "electronics"
    FOOD_BEVERAGE = "food_beverage"
    BEAUTY = "beauty_cosmetics"
    HOME_GOODS = "home_goods"
    GENERIC = "generic_b2b"


@dataclass
class PurchaseOrderLineItem:
    """Single line item on a purchase order"""
    
    # Line identification
    line_number: int
    
    # Product identification
    supplier_sku: str
    buyer_sku: Optional[str] = None
    product_name: str = ""
    
    # Product specifications
    product_attributes: Dict[str, str] = field(default_factory=dict)
    upc_code: Optional[str] = None
    
    # Quantities
    quantity_ordered: int = 1
    moq: Optional[int] = None
    unit_of_measure: str = "EA"
    
    # Pricing
    unit_cost: Decimal = Decimal("0.00")
    line_total: Decimal = Decimal("0.00")
    
    # Delivery
    requested_delivery_date: Optional[date] = None
    lead_time_days: Optional[int] = None
    
    # Metadata
    product_category: str = ""
    brand: Optional[str] = None
    weight: Optional[float] = None
    notes: Optional[str] = None
    hs_code: Optional[str] = None  # Harmonized System code for international trade
    
    def __post_init__(self):
        """Calculate line total after initialization"""
        # Convert to Decimal if not already
        if not isinstance(self.unit_cost, Decimal):
            self.unit_cost = Decimal(str(self.unit_cost))
        
        # Calculate line total
        self.line_total = self.unit_cost * Decimal(str(self.quantity_ordered))
    
    def validate(self, po_type: POType) -> List[str]:
        """
        Validate line item based on PO type requirements.
        
        Args:
            po_type: Type of purchase order
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Required fields for all types
        if not self.supplier_sku:
            errors.append(f"Line {self.line_number}: supplier_sku is required")
        if not self.product_name:
            errors.append(f"Line {self.line_number}: product_name is required")
        if self.quantity_ordered <= 0:
            errors.append(f"Line {self.line_number}: quantity_ordered must be > 0")
        if self.unit_cost <= 0:
            errors.append(f"Line {self.line_number}: unit_cost must be > 0")
        
        # Type-specific validation
        if po_type == POType.ALIBABA:
            if self.moq is None:
                errors.append(f"Line {self.line_number}: MOQ required for Alibaba PO")
            if self.lead_time_days is None:
                errors.append(f"Line {self.line_number}: lead_time required for Alibaba PO")
        
        elif po_type == POType.DROPSHIP:
            if not self.buyer_sku:
                errors.append(f"Line {self.line_number}: buyer_sku required for dropship PO")
        
        # MOQ validation
        if self.moq is not None and self.quantity_ordered < self.moq:
            errors.append(
                f"Line {self.line_number}: quantity ({self.quantity_ordered}) "
                f"is less than MOQ ({self.moq})"
            )
        
        return errors


@dataclass
class PurchaseOrder:
    """Complete purchase order document"""
    
    # PO Header
    po_number: str
    po_date: date
    status: POStatus = POStatus.DRAFT
    
    # Buyer information (your company)
    buyer_company: str = ""
    buyer_address: str = ""
    buyer_contact: str = ""
    buyer_email: str = ""
    buyer_phone: str = ""
    buyer_logo: Optional[str] = None  # Base64 encoded logo
    
    # Supplier/Vendor information
    supplier_name: str = ""
    supplier_address: str = ""
    supplier_contact: str = ""
    supplier_email: str = ""
    supplier_phone: str = ""
    supplier_account_number: Optional[str] = None
    
    # Ship-to information (usually same as buyer)
    ship_to_company: str = ""
    ship_to_address: str = ""
    ship_to_attention: str = ""
    ship_to_phone: str = ""
    
    # Line items
    line_items: List[PurchaseOrderLineItem] = field(default_factory=list)
    
    # Totals
    subtotal: Decimal = Decimal("0.00")
    shipping_cost: Decimal = Decimal("0.00")
    tax_rate: Decimal = Decimal("0.00")
    tax_amount: Decimal = Decimal("0.00")
    total_amount: Decimal = Decimal("0.00")
    
    # Terms & conditions
    payment_terms: str = "NET 30"
    shipping_method: str = "Standard Ground"
    incoterms: Optional[str] = None
    delivery_date: Optional[date] = None
    
    # Special instructions
    special_instructions: Optional[str] = None
    packing_instructions: Optional[str] = None
    
    # Metadata
    currency: str = "USD"
    po_type: POType = POType.GENERIC
    created_by: str = ""
    
    def __post_init__(self):
        """Calculate totals after initialization"""
        self.calculate_totals()
    
    def calculate_totals(self):
        """Calculate subtotal, tax, and total"""
        # Subtotal
        self.subtotal = sum(
            item.line_total for item in self.line_items
        )
        
        # Tax amount
        if not isinstance(self.tax_rate, Decimal):
            self.tax_rate = Decimal(str(self.tax_rate))
        self.tax_amount = self.subtotal * self.tax_rate
        
        # Total
        if not isinstance(self.shipping_cost, Decimal):
            self.shipping_cost = Decimal(str(self.shipping_cost))
        self.total_amount = self.subtotal + self.tax_amount + self.shipping_cost
    
    def validate(self) -> List[str]:
        """
        Validate purchase order data.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Header validation
        if not self.po_number:
            errors.append("po_number is required")
        if not self.buyer_company:
            errors.append("buyer_company is required")
        if not self.supplier_name:
            errors.append("supplier_name is required")
        if not self.ship_to_address:
            errors.append("ship_to_address is required")
        
        # Line items validation
        if not self.line_items:
            errors.append("At least one line item is required")
        
        for item in self.line_items:
            errors.extend(item.validate(self.po_type))
        
        # Type-specific validation
        if self.po_type == POType.ALIBABA:
            if not self.incoterms:
                errors.append("Incoterms required for Alibaba PO")
        
        if self.po_type == POType.DOMESTIC:
            if not self.delivery_date:
                errors.append("Delivery date required for domestic distributor PO")
        
        return errors
    
    def to_dict(self) -> dict:
        """
        Convert purchase order to dictionary for template rendering.
        
        Returns:
            Dictionary with all PO data formatted for Jinja2 templates
        """
        # Helper to parse address "123 Main St, City, ST 12345" format
        def parse_address(full_address: str) -> dict:
            """Parse address string into components"""
            parts = [p.strip() for p in full_address.split(',')]
            if len(parts) >= 3:
                # Last part should be "State ZIP"
                state_zip = parts[-1].strip().split()
                return {
                    'street': ', '.join(parts[:-2]),  # Everything before city
                    'city': parts[-2].strip(),
                    'state': state_zip[0] if state_zip else '',
                    'zip': state_zip[1] if len(state_zip) > 1 else ''
                }
            else:
                # Fallback if format doesn't match
                return {
                    'street': full_address,
                    'city': '',
                    'state': '',
                    'zip': ''
                }
        
        # Parse addresses
        buyer_addr = parse_address(self.buyer_address) if self.buyer_address else {}
        supplier_addr = parse_address(self.supplier_address) if self.supplier_address else {}
        ship_to_addr = parse_address(self.ship_to_address) if self.ship_to_address else {}
        
        return {
            # Header
            'po_number': self.po_number,
            'po_date': self.po_date.strftime('%B %d, %Y') if isinstance(self.po_date, date) else self.po_date,
            'status': self.status.value if isinstance(self.status, POStatus) else self.status,
            'requested_delivery_date': self.delivery_date.strftime('%B %d, %Y') if self.delivery_date else None,
            
            # Buyer
            'buyer_company': self.buyer_company,
            'buyer_address': buyer_addr.get('street', self.buyer_address),
            'buyer_city': buyer_addr.get('city', ''),
            'buyer_state': buyer_addr.get('state', ''),
            'buyer_zip': buyer_addr.get('zip', ''),
            'buyer_contact': self.buyer_contact,
            'buyer_contact_name': self.buyer_contact,  # Alias for signatures
            'buyer_email': self.buyer_email,
            'buyer_phone': self.buyer_phone,
            'buyer_logo': self.buyer_logo,
            
            # Supplier
            'supplier_name': self.supplier_name,
            'supplier_address': supplier_addr.get('street', self.supplier_address),
            'supplier_city': supplier_addr.get('city', ''),
            'supplier_state': supplier_addr.get('state', ''),
            'supplier_zip': supplier_addr.get('zip', ''),
            'supplier_contact': self.supplier_contact,
            'supplier_email': self.supplier_email,
            'supplier_phone': self.supplier_phone,
            'supplier_account_number': self.supplier_account_number,
            
            # Ship-to
            'ship_to_name': self.ship_to_company or self.buyer_company,  # Default to buyer
            'ship_to_company': self.ship_to_company,
            'ship_to_address': ship_to_addr.get('street', self.ship_to_address),
            'ship_to_city': ship_to_addr.get('city', ''),
            'ship_to_state': ship_to_addr.get('state', ''),
            'ship_to_zip': ship_to_addr.get('zip', ''),
            'ship_to_attention': self.ship_to_attention,
            'ship_to_phone': self.ship_to_phone,
            
            # Items (for template rendering - use 'line_items' for compatibility)
            'line_items': [
                {
                    'line_number': item.line_number,
                    'supplier_sku': item.supplier_sku,
                    'buyer_sku': item.buyer_sku,
                    'description': item.product_name,
                    'product_name': item.product_name,
                    'quantity': item.quantity_ordered,
                    'quantity_ordered': item.quantity_ordered,
                    'moq': item.moq,
                    'unit_of_measure': item.unit_of_measure,
                    'unit_cost': float(item.unit_cost),
                    'unit_price': float(item.unit_cost),  # Alias
                    'total': float(item.line_total),
                    'line_total': float(item.line_total),
                    'product_attributes': item.product_attributes,
                    'product_category': item.product_category,
                    'upc_code': item.upc_code,
                    'brand': item.brand,
                    'lead_time_days': item.lead_time_days,
                    'hs_code': item.hs_code,
                    'requested_delivery_date': item.requested_delivery_date.strftime('%B %d, %Y') if item.requested_delivery_date else None,
                    'notes': item.notes,
                }
                for item in self.line_items
            ],
            
            # Totals
            'subtotal': float(self.subtotal),
            'shipping_cost': float(self.shipping_cost),
            'tax_rate': float(self.tax_rate),
            'tax_amount': float(self.tax_amount),
            'total_amount': float(self.total_amount),
            
            # Terms
            'payment_terms': self.payment_terms,
            'shipping_method': self.shipping_method,
            'incoterms': self.incoterms,
            'delivery_date': self.delivery_date.strftime('%B %d, %Y') if self.delivery_date else None,
            
            # Instructions
            'special_instructions': self.special_instructions,
            'packing_instructions': self.packing_instructions,
            
            # Metadata
            'currency': self.currency,
            'po_type': self.po_type.value if isinstance(self.po_type, POType) else self.po_type,
            'created_by': self.created_by,
            
            # Convenience fields
            'item_count': len(self.line_items),
        }


def create_sample_po() -> PurchaseOrder:
    """Create a sample purchase order for testing"""
    
    line_items = [
        PurchaseOrderLineItem(
            line_number=1,
            supplier_sku="WGT-PRO-500",
            buyer_sku="PROD-001",
            product_name="Professional Widget - Model 500",
            quantity_ordered=100,
            unit_of_measure="EA",
            unit_cost=Decimal("12.50"),
            moq=50,
            lead_time_days=30,
            product_category="Electronics",
            brand="WidgetPro",
        ),
        PurchaseOrderLineItem(
            line_number=2,
            supplier_sku="GDT-MAX-1000",
            buyer_sku="PROD-002",
            product_name="Gadget Max - Premium Edition",
            quantity_ordered=50,
            unit_of_measure="EA",
            unit_cost=Decimal("24.00"),
            moq=25,
            lead_time_days=45,
            product_category="Electronics",
            brand="GadgetMax",
        ),
    ]
    
    po = PurchaseOrder(
        po_number="PO-2024-12345",
        po_date=date.today(),
        status=POStatus.DRAFT,
        
        buyer_company="Your E-Commerce Store",
        buyer_address="123 Main Street, Suite 100, New York, NY 10001",
        buyer_contact="John Smith",
        buyer_email="john.smith@yourstore.com",
        buyer_phone="(555) 123-4567",
        
        supplier_name="Global Wholesale Distributors",
        supplier_address="456 Industrial Blvd, Los Angeles, CA 90001",
        supplier_contact="Jane Doe",
        supplier_email="jane.doe@globalwholesale.com",
        supplier_phone="(555) 987-6543",
        supplier_account_number="ACCT-98765",
        
        ship_to_company="Your E-Commerce Store - Warehouse",
        ship_to_address="789 Warehouse Drive, Newark, NJ 07101",
        ship_to_attention="Receiving Department",
        ship_to_phone="(555) 111-2222",
        
        line_items=line_items,
        
        shipping_cost=Decimal("150.00"),
        tax_rate=Decimal("0.08"),
        
        payment_terms="NET 30",
        shipping_method="FOB Origin",
        delivery_date=date(2025, 1, 15),
        
        special_instructions="Please notify before shipping. Pack items in original manufacturer boxes.",
        
        currency="USD",
        po_type=POType.DOMESTIC,
        created_by="John Smith",
    )
    
    return po


if __name__ == "__main__":
    # Test the data model
    po = create_sample_po()
    
    # Validate
    errors = po.validate()
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Purchase order is valid")
    
    # Print summary
    print(f"\nPurchase Order: {po.po_number}")
    print(f"Supplier: {po.supplier_name}")
    print(f"Items: {len(po.line_items)}")
    print(f"Subtotal: ${po.subtotal}")
    print(f"Total: ${po.total_amount}")
    
    # Test to_dict conversion
    data = po.to_dict()
    print(f"\n✓ Converted to dict with {len(data)} fields")
    print(f"✓ Items array has {len(data['items'])} items")
