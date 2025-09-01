from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union
from decimal import Decimal
from datetime import datetime
from enum import Enum
import json

class CurrencyEnum(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"

class CoverageTypeEnum(str, Enum):
    LIABILITY = "liability"
    PROPERTY = "property"
    AUTO = "auto"
    WORKERS_COMP = "workers_compensation"
    PROFESSIONAL = "professional_liability"
    CYBER = "cyber_liability"
    UMBRELLA = "umbrella"
    DIRECTORS_OFFICERS = "directors_and_officers"
    OTHER = "other"

class LimitTypeEnum(str, Enum):
    PER_OCCURRENCE = "per_occurrence"
    AGGREGATE = "aggregate"
    PER_CLAIM = "per_claim"
    COMBINED_SINGLE = "combined_single_limit"
    SPLIT = "split_limit"
    SUBLIMIT = "sublimit"

class Amount(BaseModel):
    """Monetary amount with currency"""
    value: Optional[Union[Decimal, float]] = Field(
        None, 
        description="Numerical amount (can be percentage like 2.624 for 2.624% or dollar amount like 1000000)"
    )
    currency: Optional[CurrencyEnum] = Field(
        None, 
        description="Currency code (null for percentages, required for monetary amounts)"
    )
    formatted_text: Optional[str] = Field(
        None, 
        description="Original text representation from document (e.g., '$1,000,000', '2.624%', '£500,000')"
    )
    
    @validator('currency', pre=True)
    def normalize_currency(cls, v):
        """Normalize currency values, converting N/A to None for percentages"""
        if v in ['N/A', 'n/a', 'null', '']:
            return None
        return v

class UniqueMarketReference(BaseModel):
    """Unique identifiers for the insurance contract"""
    policy_number: Optional[str] = Field(None, description="Primary policy number")
    quote_number: Optional[str] = Field(None, description="Quote reference number")
    broker_reference: Optional[str] = Field(None, description="Broker's reference number")
    insurer_reference: Optional[str] = Field(None, description="Insurer's internal reference")
    umr_code: Optional[str] = Field(None, description="Unique Market Reference code")
    certificate_number: Optional[str] = Field(None, description="Certificate number if applicable")
    endorsement_numbers: List[str] = Field(default_factory=list, description="List of endorsement numbers")

class Limit(BaseModel):
    """Insurance coverage limits"""
    limit_type: LimitTypeEnum = Field(..., description="Type of limit")
    amount: Amount = Field(..., description="Limit amount")
    description: str = Field(..., description="Description of what this limit applies to")
    coverage_reference: Optional[str] = Field(None, description="Which coverage this limit applies to")
    deductible: Optional[Amount] = Field(None, description="Associated deductible amount")
    waiting_period: Optional[str] = Field(None, description="Waiting period if applicable")
    
    @validator('limit_type', pre=True)
    def normalize_limit_type(cls, v):
        """Normalize limit type to standard enum values"""
        if isinstance(v, str):
            v_lower = v.lower()
            if 'occurrence' in v_lower or 'catastrophe' in v_lower or 'excess' in v_lower:
                return LimitTypeEnum.PER_OCCURRENCE
            elif 'aggregate' in v_lower:
                return LimitTypeEnum.AGGREGATE
            elif 'claim' in v_lower:
                return LimitTypeEnum.PER_CLAIM
            elif 'combined' in v_lower or 'single' in v_lower:
                return LimitTypeEnum.COMBINED_SINGLE
            elif 'split' in v_lower:
                return LimitTypeEnum.SPLIT
            elif 'sublimit' in v_lower or 'sub' in v_lower:
                return LimitTypeEnum.SUBLIMIT
        return v

class Premium(BaseModel):
    """Premium information with detailed extraction guidance"""
    premium_type: str = Field(
        ..., 
        description="Type of premium: annual_premium, deposit_premium, minimum_premium, quarterly_deposit, rate_percentage, installment_premium, total_premium, base_premium"
    )
    amount: Amount = Field(
        ..., 
        description="Premium amount - can be percentage rates (e.g., 2.624%), flat amounts (e.g., $3,600,000), or formulas"
    )
    coverage_reference: Optional[str] = Field(
        None, 
        description="Coverage layer reference, accounting code, or part identifier (e.g., 'POR1008712', 'First Catastrophe', 'Layer 1')"
    )
    payment_frequency: Optional[str] = Field(
        None, 
        description="Payment schedule: annual, quarterly, monthly, installments, semi-annual"
    )
    due_date: Optional[datetime] = Field(None, description="When premium payment is due (ISO date format)")
    taxes_and_fees: Optional[Amount] = Field(None, description="Additional taxes, fees, and surcharges")
    
    @validator('premium_type')
    def normalize_premium_type(cls, v):
        """Normalize premium type to standard values"""
        v_lower = v.lower().replace(' ', '_')
        type_mapping = {
            'deposit_premium': 'deposit_premium',
            'minimum_premium': 'minimum_premium', 
            'rate_percentage': 'rate_percentage',
            'annual_premium': 'annual_premium',
            'quarterly_deposit': 'quarterly_deposit'
        }
        return type_mapping.get(v_lower, v)

class Coverage(BaseModel):
    """Insurance coverage details"""
    coverage_id: Optional[str] = Field(None, description="Internal identifier for this coverage")
    coverage_type: CoverageTypeEnum = Field(..., description="Type of coverage")
    coverage_name: str = Field(..., description="Name or title of the coverage")
    description: str = Field(..., description="Detailed description of what is covered")
    effective_date: Optional[datetime] = Field(None, description="When coverage begins")
    expiration_date: Optional[datetime] = Field(None, description="When coverage ends")
    territory: Optional[str] = Field(None, description="Geographic territory covered")
    conditions: List[str] = Field(default_factory=list, description="Specific conditions for this coverage")
    extensions: List[str] = Field(default_factory=list, description="Coverage extensions or endorsements")

class Exclusion(BaseModel):
    """Exclusions from coverage with detailed extraction patterns"""
    exclusion_id: Optional[str] = Field(None, description="Internal identifier for this exclusion")
    exclusion_type: str = Field(
        ..., 
        description="Category: war_exclusion, nuclear_exclusion, flood_exclusion, terrorism_exclusion, general_exclusion, business_exclusion, deductible_exclusion, insolvency_exclusion, pool_exclusion"
    )
    title: str = Field(
        ..., 
        description="Title from document (e.g., 'War and Military Action', 'Nuclear Incident Exclusion', 'High Deductible Policies')"
    )
    description: str = Field(
        ..., 
        description="What is excluded - look for language like 'does not cover', 'shall not apply', 'excluded', 'not covered', 'this agreement does not'"
    )
    applies_to_coverage: List[str] = Field(
        default_factory=list, 
        description="Which specific coverages or layers this exclusion affects"
    )
    exceptions: List[str] = Field(
        default_factory=list, 
        description="Exceptions to exclusion - phrases like 'except', 'provided this exclusion shall not apply', 'this shall not operate'"
    )
    cross_references: List[str] = Field(
        default_factory=list, 
        description="References to clauses, articles, or attached schedules mentioned in exclusion text"
    )

class InsuranceContract(BaseModel):
    """Complete insurance contract extraction model"""
    unique_market_reference: UniqueMarketReference = Field(..., description="All unique identifiers for this contract")
    limits: List[Limit] = Field(default_factory=list, description="All coverage limits")
    premiums: List[Premium] = Field(default_factory=list, description="All premium information")
    coverages: List[Coverage] = Field(default_factory=list, description="All coverage details")
    exclusions: List[Exclusion] = Field(default_factory=list, description="All exclusions")
    
    # Additional metadata
    extraction_metadata: Optional[Dict[str, str]] = Field(
        default_factory=dict, 
        description="Metadata about the extraction process"
    )

    @validator('limits')
    def validate_limits_have_amounts(cls, v):
        """Ensure all limits have valid amounts when present"""
        for limit in v:
            if limit.amount.value is not None and limit.amount.value <= 0:
                raise ValueError(f"Limit amount must be positive: {limit.description}")
        return v

    @validator('premiums')
    def validate_premiums_positive(cls, v):
        """Ensure all premiums are positive when present"""
        for premium in v:
            if premium.amount.value is not None and premium.amount.value <= 0:
                raise ValueError(f"Premium amount must be positive: {premium.premium_type}")
        return v

    def get_total_premium(self) -> Optional[Amount]:
        """Helper method to get total premium if available"""
        for premium in self.premiums:
            if premium.premium_type.lower() in ['total', 'total_premium', 'grand_total']:
                return premium.amount
        return None

    def get_coverage_by_type(self, coverage_type: CoverageTypeEnum) -> List[Coverage]:
        """Helper method to get coverages by type"""
        return [c for c in self.coverages if c.coverage_type == coverage_type]

    def get_limits_for_coverage(self, coverage_id: str) -> List[Limit]:
        """Helper method to get limits for specific coverage"""
        return [l for l in self.limits if l.coverage_reference == coverage_id]
    
    @classmethod
    def get_extraction_schema(cls) -> Dict:
        """Generate JSON schema with extraction guidance for LLM prompts"""
        schema = cls.schema()
        
        # Add extraction guidance to the schema
        if 'properties' in schema:
            # Add premium extraction patterns
            if 'premiums' in schema['properties']:
                schema['properties']['premiums']['extraction_patterns'] = {
                    "look_for_terms": ["premium", "rate", "deposit", "minimum premium", "annual premium", "quarterly", "installment"],
                    "table_sections": ["Premium", "Reinsurance Premium", "Payment Terms", "Rate Schedule"],
                    "rate_patterns": ["2.624%", "3.032%", "5.832%", "rate percentage"],
                    "amount_patterns": ["$3,600,000", "£500,000", "deposit premium", "minimum premium"],
                    "payment_patterns": ["quarterly deposits", "four equal installments", "annual", "monthly"],
                    "reference_patterns": ["POR1008712", "First Catastrophe", "Layer 1", "Part I", "coverage code"]
                }
            
            # Add exclusion extraction patterns  
            if 'exclusions' in schema['properties']:
                schema['properties']['exclusions']['extraction_patterns'] = {
                    "section_headers": ["EXCLUSIONS", "THIS AGREEMENT DOES NOT COVER", "ARTICLE IX", "General Exclusions", "Standard Exclusions"],
                    "exclusion_language": ["does not cover", "shall not apply", "excluded", "not covered", "this agreement does not", "shall not", "does not apply"],
                    "exclusion_types": ["war", "nuclear", "terrorism", "flood", "earthquake", "pollution", "cyber", "deductible", "insolvency", "pool", "syndicate"],
                    "exception_language": ["except", "provided this exclusion shall not apply", "this shall not operate", "however", "but not excluding"],
                    "common_exclusions": ["war exclusion", "nuclear incident exclusion", "high deductible policies", "pool and syndicate business", "insolvency funds"]
                }
        
        return schema
    
    @classmethod 
    def get_extraction_instructions(cls) -> str:
        """Generate detailed extraction instructions for LLM"""
        return """
EXTRACTION INSTRUCTIONS:

PREMIUM EXTRACTION:
- Look for sections titled: Premium, Reinsurance Premium, Payment Terms, Rate Schedule
- Premium types include: annual_premium, deposit_premium, minimum_premium, quarterly_deposit, rate_percentage
- Rates can be percentages (2.624%) or dollar amounts ($3,600,000)
- Each coverage layer/part may have separate premium structures
- Payment frequencies: annual, quarterly, monthly, installments
- Create separate entries for different premium types (rate, minimum, deposit) even for same coverage
- Look for coverage references like POR1008712, First Catastrophe, Layer 1

EXCLUSION EXTRACTION:
- Look for sections titled: EXCLUSIONS, THIS AGREEMENT DOES NOT COVER, General Exclusions
- Exclusion language includes: 'does not cover', 'shall not apply', 'excluded', 'not covered'
- Common exclusion types: war_exclusion, nuclear_exclusion, terrorism_exclusion, flood_exclusion
- Pay attention to exceptions using words like: 'except', 'provided this exclusion shall not apply'
- Extract cross-references to other policy sections or attached clauses
- Group related exclusions under appropriate types (war, nuclear, business, etc.)

GENERAL GUIDANCE:
- Extract only explicit information from the document
- Use null for missing data
- For dates use YYYY-MM-DD format
- Be concise in descriptions (max 2 sentences each)
- Return valid JSON only
"""

# Example usage and validation
# if __name__ == "__main__":
#     # Example of how to create an instance
#     sample_contract = InsuranceContract(
#         unique_market_reference=UniqueMarketReference(
#             policy_number="POL-2024-001234",
#             umr_code="UMR-ABC123"
#         ),
#         limits=[
#             Limit(
#                 limit_type=LimitTypeEnum.PER_OCCURRENCE,
#                 amount=Amount(value=1000000, currency=CurrencyEnum.USD),
#                 description="General Liability Per Occurrence Limit"
#             )
#         ],
#         premiums=[
#             Premium(
#                 premium_type="annual_total",
#                 amount=Amount(value=15000, currency=CurrencyEnum.USD)
#             )
#         ],
#         coverages=[
#             Coverage(
#                 coverage_type=CoverageTypeEnum.LIABILITY,
#                 coverage_name="General Liability",
#                 description="Coverage for bodily injury and property damage claims"
#             )
#         ],
#         exclusions=[
#             Exclusion(
#                 exclusion_type="standard",
#                 title="War and Military Action",
#                 description="Coverage does not apply to war, invasion, or military action"
#             )
#         ]
#     )
    
#     print("Sample contract created successfully!")
#     print(f"Policy Number: {sample_contract.unique_market_reference.policy_number}")
#     print(f"Total Coverages: {len(sample_contract.coverages)}")
#     print(f"Total Exclusions: {len(sample_contract.exclusions)}")