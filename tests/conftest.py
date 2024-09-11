import os
import shutil
import pytest
from pymongo import MongoClient
from pymongo.database import Database
from typing import Generator

@pytest.fixture
def mongo_client():
    try:    
        mongo_client = MongoClient('mongodb://localhost:27017')
        yield mongo_client
    finally:
        mongo_client.drop_database('tests')
        mongo_client.close()


@pytest.fixture
def mongo_database(mongo_client: MongoClient) -> Generator[Database, None, None]:
    yield mongo_client['tests']
    
@pytest.fixture(scope='session')
def directory():
    if os.path.exists('data/tests/weights'):
        shutil.rmtree('data/tests/weights')
    yield 'data/tests/weights'
    shutil.rmtree('data/tests/weights')