import logging
from sqlalchemy.engine import Connectable
from sqlalchemy.orm import Session
from app.db.model import Evaluate
from app.setting.exceptions import DBInsertError


logger = logging.getLogger(__name__)

def insert_evaluation(obj: dict, con: Connectable):
    """
    Insert evaluation to postgre sql db
    """
    evaluation = Evaluate(**obj)
    with Session(con) as session:
        session.begin()
        retries = 3
        while retries > 0:
            try:
                session.add(evaluation)
                session.commit()
                logger.info("Insert evaluation success.", extra={obj})
                break
            except Exception as e:
                session.rollback()
                logger.info("Insert evaluation fail, retry ...", extra={obj})
                retries -= 1
                if retries == 0:
                    logger.exception("Retry times are exhausted, insert log fail! ", extra={obj})
                    raise DBInsertError(e)
            finally:
                session.close()
