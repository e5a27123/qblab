import os
import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException
from app.schemas.payload import ChatRequest, GenaiRequest, EvaluateRequest
from app.schemas.response import ChatResponse, GenaiResponse, EvaluateResponse, MWHeader, ChatTranRS, GenAIResponseTranRS, EvaluateTranRS, GenAIModel, BaseGenaiTemplate, ChatTemplate
from app.query_vdb import ChromaDBClient
from utils import mlflow_exception_logger, mlflow_openai_callback

app = FastAPI()


# 定义路由
@app.post("/api/card-consumption/chat", response_model=ChatResponse)
@mlflow_exception_logger
@mlflow_openai_callback
async def chat(request: ChatRequest):
    # 在这里处理聊天请求
    mwheader = MWHeader(MSGID=request.MWHEADER.MSGID,
                        SOURCECHANNEL=request.MWHEADER.SOURCECHANNEL,
                        TXNSEQ=request.MWHEADER.TXNSEQ,
                        RETURNCODE="0000",
                        RETURNDESC="成功"
    )
    collection_name = 'collect_cubelab_qa_lite'
    chroma_client = ChromaDBClient(collection_name=collection_name)
    result = chroma_client.query(query_texts="test123", n_results=1)
    print(result)
    # 建立 ChatTranRS 回應
    tranrs = ChatTranRS(
        sessionId=request.TRANRQ.sessionId,
        customerId=request.TRANRQ.customerId,
        template=ChatTemplate(
            tid="01",
            blockReason=None,
            startDate="2024-05-11",
            endDate="2024-06-11",
            storeName=["信義微風"],
            categoryName=["百貨類別"]
        )
    )

    # 建立 ChatResponse 回應
    response = ChatResponse(
        MWHEADER=mwheader,
        TRANRS=tranrs
    )
    return response

@app.post("/api/card-consumption/genai-response", response_model=GenaiResponse)
@mlflow_exception_logger
@mlflow_openai_callback
async def gai_response(request: GenaiRequest):
    mwheader = MWHeader(MSGID=request.MWHEADER.MSGID,
                        SOURCECHANNEL=request.MWHEADER.SOURCECHANNEL,
                        TXNSEQ=request.MWHEADER.TXNSEQ,
                        RETURNCODE="0000",
                        RETURNDESC="成功"
    )

    tranrs = GenAIResponseTranRS(
        sessionId=request.TRANRQ.sessionId,
        customerId=request.TRANRQ.customerId,
        genAI=GenAIModel(message="mother fucker"),
        template=BaseGenaiTemplate(tid="01",
                                   blockReason=None)
    )
    # 建立 ChatResponse 回應
    response = GenaiResponse(
        MWHEADER=mwheader,
        TRANRS=tranrs
    )
    return response

@app.post("/api/card-consumption/evaluate", response_model=EvaluateResponse)
@mlflow_exception_logger
@mlflow_openai_callback
async def evaluate(request: EvaluateRequest):
    mwheader = MWHeader(MSGID=request.MWHEADER.MSGID,
                        SOURCECHANNEL=request.MWHEADER.SOURCECHANNEL,
                        TXNSEQ=request.MWHEADER.TXNSEQ,
                        RETURNCODE="0000",
                        RETURNDESC="成功"
    )
    print(request)
    # 建立 ChatTranRS 回應
    tranrs = EvaluateTranRS(
        sessionId=request.TRANRQ.sessionId,
        customerId=request.TRANRQ.customerId
    )

    # 建立 ChatResponse 回應
    response = EvaluateResponse(
        MWHEADER=mwheader,
        TRANRS=tranrs
    )
    return response

# 运行应用
if __name__ == "__main__":
    os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "30"
    tracking_server_uri = os.environ["TRACKING_SERVER_URI"]
    # tracking_server_uri="http://0.0.0.0:5000/"
    mlflow.set_tracking_uri(tracking_server_uri)
    mlflow.set_experiment("cubelab_demo")
    mlflow.langchain.autolog()
    with mlflow.start_run() as run:
        uvicorn.run(app="main:app", host="0.0.0.0", port=8080)