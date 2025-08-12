from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List

class ExtractRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded receipt image (PNG/JPG)")
    reference: Optional[str] = None

class ExtractedData(BaseModel):
    MerchantName: Optional[str] = None
    MerchantAddress: Optional[str] = None
    TransactionDate: Optional[str] = None
    StoreID: Optional[str] = None
    InvoiceId: Optional[str] = None
    CR: Optional[str] = None
    TaxID: Optional[str] = None
    Subtotal: Optional[float] = None
    Tax: Optional[float] = None
    Total: Optional[float] = None

class ValidationResult(BaseModel):
    fraudScore: int
    confidenceScore: int
    checks: Dict[str, Any]
    issues: List[str] = []

class AnalyzeResponse(BaseModel):
    data: ExtractedData
    validation: ValidationResult
    reason: str
    usage: Dict[str, Any]

class PromptDoc(BaseModel):
    _id: Optional[Any]
    scope: str = "global"  # allow future per-tenant/per-venue prompts
    system_prompt: str
    version: int
    is_active: bool = True

class SetPromptRequest(BaseModel):
    system_prompt: str