# response.py

from pydantic import BaseModel, constr
from typing import Optional, List, Dict

# MWHEADER 模型 for response
class MWHeader(BaseModel):
    MSGID: Optional[constr(max_length=20)]
    SOURCECHANNEL: constr(max_length=20)
    TXNSEQ: constr(max_length=50)
    RETURNCODE: Optional[constr(max_length=4)] = None
    RETURNDESC: Optional[constr(max_length=50)] = None
    ERRORHISTORY: Optional[constr(max_length=50)] = None
    O360SEQ: Optional[constr(max_length=36)] = None

    class Config:
        validate_assignment = True


# Base gai template field
class BaseGenaiTemplate(BaseModel):
    tid: constr(max_length=2)
    blockReason: Optional[constr(max_length=2)]


# TranRS template field for /chat response
class ChatTemplate(BaseGenaiTemplate):
    startDate: constr(max_length=10)
    endDate: constr(max_length=10)
    storeName: Optional[List[str]] = None
    categoryName: Optional[List[str]]= None

    class Config:
        validate_assignment = True


# TranRS for /chat
class ChatTranRS(BaseModel):
    sessionId: constr(max_length=12)
    customerId: constr(max_length=10)
    template: ChatTemplate

# TranRS for GenAI field
class GenAIModel(BaseModel):
    message: str

    class Config:
        validate_assignment = True


# TranRS for /genai-response
class GenAIResponseTranRS(BaseModel):
    sessionId: constr(max_length=12)
    customerId: constr(max_length=10)
    genAI: GenAIModel
    template: BaseGenaiTemplate


# TRANRS for /evaluate response
class EvaluateTranRS(BaseModel):
    sessionId: constr(max_length=12)
    customerId: constr(max_length=10)


# response model for /chat
class ChatResponse(BaseModel):
    MWHEADER: MWHeader
    TRANRS: ChatTranRS

    class Config:
        validate_assignment = True


# response model for /genai-response
class GenaiResponse(BaseModel):
    MWHEADER: MWHeader
    TRANRS: GenAIResponseTranRS

    class Config:
        validate_assignment = True


# response model for /evaluate
class EvaluateResponse(BaseModel):
    MWHEADER: MWHeader
    TRANRS: EvaluateTranRS

    class Config:
        validate_assignment = True

