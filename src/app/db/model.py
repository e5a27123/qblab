from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class Evaluate(Base):
    __tablename__ = 'evaluate'
    TXNSEQ = Column(String, primary_key=True)
    SESSIONI_ID = Column(String, primary_key=True)
    CUSTOMER_ID = Column(String, primary_key=True)
    EVALUATE = Column(Boolean)
    TIME = Column(DateTime)