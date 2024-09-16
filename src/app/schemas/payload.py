
from pydantic import BaseModel, constr, conlist, conint, condecimal
from typing import Optional, List

# MWHEADER 模型
class MWHeader(BaseModel):
    MSGID: Optional[constr(max_length=20)]
    SOURCECHANNEL: constr(max_length=20)
    TXNSEQ: constr(max_length=50)


class BaseTRANRQ(BaseModel):
    sessionId: constr(max_length=12)
    customerId: constr(max_length=10)
    time: constr(max_length=20)


# TRANRQ 模型 for /chat request
class ChatTRANRQ(BaseTRANRQ):
    message: str

# TRANRQ 模型 for /genai-response request
class GenaiTranRQ(BaseTRANRQ):
    tid: constr(max_length=2)
    consumptionNumber: conint(ge=0)
    totalAmount: condecimal(max_digits=20, decimal_places=2)
    startDate: constr(max_length=10)
    endDate: constr(max_length=10)
    storeName: Optional[List[str]] = None
    categoryName: Optional[List[str]] = None

# TRANRQ 模型 for /evaluate request
class EvaluateTranRQ(BaseTRANRQ):
    evaluate: bool

# request model for /chat
class ChatRequest(BaseModel):
    """
    {
        "MWHEADER": {
            "MSGID": null,
            "SOURCECHANNEL": "CHAT-API",
            "TXNSEQ": "008788888-da-aaa-dd"
        },
        "TRANRQ": {
            "sessionId": "16574823aA",
            "customerId": "E222222897",
            "message": "你好，你是什麼東西",
            "time": "2024/09/02 15:35:40"
        }
    }
    """    
    MWHEADER: MWHeader
    TRANRQ: ChatTRANRQ

# request model for /genai-response
class GenaiRequest(BaseModel):
    """
    {
        "MWHEADER": {
            "MSGID": null,
            "SOURCECHANNEL": "CHAT-API",
            "TXNSEQ": "008788888-da-aaa-dd"
        },
        "TRANRQ": {
            "sessionId": "16574823aA",
            "customerId": "E222222897",
            "tid": "99",
            "consumptionNumber": 12,
            "totalAmount": 24000.38,
            "startDate": "2024/05/11",
            "endDate": "2024/06/11",
            "storeName": ["信義微風"],
            "categoryName": ["百貨類別"],
            "time": "2024/09/02 16:35:40"
        }
    }
    """
    MWHEADER: MWHeader
    TRANRQ: GenaiTranRQ

# request model for /genai-response
class EvaluateRequest(BaseModel):
    """
    {
        "MWHEADER": {
            "MSGID": null,
            "SOURCECHANNEL": "CHAT-API",
            "TXNSEQ": "008788888-da-aaa-dd"
        },
        "TRANRQ": {
            "sessionId": "16574823aA",
            "customerId": "E222222897",
            "evaluate": true,
            "time": "2024/09/02 15:35:40"
        }
    }
    """
    MWHEADER: MWHeader
    TRANRQ: EvaluateTranRQ

