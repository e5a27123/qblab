# response.py

from pydantic import BaseModel, constr
from typing import Optional, List, Dict

# MWHEADER 模型 for response
class MWHeader(BaseModel):
    MSGID: Optional[constr(max_length=20)] = None
    SOURCECHANNEL: constr(max_length=20) = None
    TXNSEQ: constr(max_length=50) = None
    RETURNCODE: Optional[constr(max_length=4)] = "0000"
    RETURNDESC: Optional[constr(max_length=50)] = "成功"
    ERRORHISTORY: Optional[constr(max_length=50)] = None
    O360SEQ: Optional[constr(max_length=36)] = None

    class Config:
        validate_assignment = True


# Base gai template field
class BaseGenaiTemplate(BaseModel):
    tid: constr(max_length=2) = None
    blockReason: Optional[str] = None


# TranRS template field for /chat response
class ChatTemplate(BaseGenaiTemplate):
    startDate: Optional[constr(max_length=10)] = None
    endDate: Optional[constr(max_length=10)] = None
    storeName: List[Optional[str]] = None
    categoryName: List[Optional[str]] = None

    class Config:
        validate_assignment = True


# TranRS for /chat
class ChatTranRS(BaseModel):
    sessionId: constr(max_length=12) = None
    customerId: constr(max_length=10) = None
    template: ChatTemplate = ChatTemplate()

# TranRS for GenAI field
class GenAIModel(BaseModel):
    message: str = None

    class Config:
        validate_assignment = True


# TranRS for /genai-response
class GenAIResponseTranRS(BaseModel):
    sessionId: constr(max_length=12) = None
    customerId: constr(max_length=10) = None
    genAI: GenAIModel = GenAIModel()
    template: BaseGenaiTemplate = BaseGenaiTemplate()


# TRANRS for /evaluate response
class EvaluateTranRS(BaseModel):
    sessionId: constr(max_length=12) = None
    customerId: constr(max_length=10) = None


# response model for /chat
class ChatResponse(BaseModel):
    MWHEADER: MWHeader = MWHeader()
    TRANRS: ChatTranRS = ChatTranRS()

    class Config:
        validate_assignment = True


# response model for /genai-response
class GenaiResponse(BaseModel):
    MWHEADER: MWHeader = MWHeader()
    TRANRS: GenAIResponseTranRS = GenAIResponseTranRS()

    class Config:
        validate_assignment = True


# response model for /evaluate
class EvaluateResponse(BaseModel):
    MWHEADER: MWHeader = MWHeader()
    TRANRS: EvaluateTranRS = EvaluateTranRS()

    class Config:
        validate_assignment = True

