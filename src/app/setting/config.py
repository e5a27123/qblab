import os

class Settings():

    VDB_COLLECTION = "collect_cubelab_qa_lite"
    DB_USER = os.environ["DB_USER"]
    DB_PWD = os.environ["DB_PWD"]
    DB_DSN = os.environ["DB_DSN"]
    ENGINE_ECHO = True

