import pytest
from pymongo import MongoClient
from pymongo.database import Database
from typing import Generator

@pytest.fixture
def mongo_client():
    return MongoClient('mongodb://localhost:27017')

@pytest.fixture
def mongo_database(mongo_client: MongoClient) -> Generator[Database, None, None]:
    yield mongo_client['test']
    mongo_client.drop_database('test')