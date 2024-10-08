import os
from typing import Dict, List
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from setting.constant import PROMPT_GENAI_RESPONSE, CUST_DESC

class GenAIResponse:

    def __init__(self, sessionId, customerId):
        self.sessionId = sessionId
        self.customerId = customerId
        self.tone = self._set_tone()
        self.chain = self._create_chain_response()

    def _create_chain_response(self):

        model = AzureChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        )
        prompt = ChatPromptTemplate.from_template(PROMPT_GENAI_RESPONSE)
        parser = StrOutputParser()

        chain = prompt | model | parser
        return chain

    def _set_tone(self):
        if self.customerId == "A":
            return "高端VIP用戶"
        elif self.customerId == "B":
            return "重點關心客戶"
        else:
            return "一般用戶"

    def generate_answer(
        self,
        tid: str,
        message: str,
        consumptionNumber: str,
        totalAmount: str,
        storeName: List,
        categoryName: List,
        **kwargs,
    ) -> Dict:

        response = {}
        response["genAI"] = {"message": None}
        response["template"] = {"tid": None, "blockReason": None}

        try:
            if "被阻擋" in (
                gen_ai_message := self.chain.invoke(
                    {
                        "message": message,
                        "consumptionNumber": consumptionNumber,
                        "totalAmount": totalAmount,
                        "storeName": storeName,
                        "categoryName": categoryName,
                        "tone": self.tone,
                        "desc": CUST_DESC.get(self.tone),
                    }
                )
            ):
                response["template"]["tid"] = "98"
                response["template"]["blockReason"] = gen_ai_message

            else:

                response["genAI"]["message"] = gen_ai_message
                response["template"]["tid"] = tid
                response["template"]["blockReason"] = None

        except Exception as e:
            response["template"]["tid"] = "98"
            response["template"]["blockReason"] = e

        finally:
            response["sessionId"] = self.sessionId
            response["customerId"] = self.customerId

        return response

