from setting.config import Settings as settings
from sqlalchemy import create_engine

sqlalchemy_engine = create_engine(
    f"postgresql://{settings.DB_USER}:{settings.DB_PWD}@{settings.DB_DSN}",
    echo=settings.ENGINE_ECHO
)