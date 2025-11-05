from __future__ import annotations

import os

from pymongo.mongo_client import MongoClient

from .logger import logger


class MongodbService:
    _instance = None
    _initialized = False

    def __new__(cls,
        collections_mapping: Dict[str, str] = None
    ) -> MongodbService:
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(MongodbService, cls).__new__(cls)
            cls._instance.__init__(collections_mapping)
        return cls._instance

    @classmethod
    def create_custom_instance(cls,
        collections_mapping: Dict[str, str] = None
    ) -> MongodbService:
        """Create a new instance with custom collections mapping, bypassing the singleton pattern."""
        instance = super(MongodbService, cls).__new__(cls)
        instance.__init__(collections_mapping)
        return instance

    def __init__(self,
        collections_mapping: Dict[str, str] = None,
        self_hosted: bool = True 
        # If you use online, don't forget to set this to False for added security
    ):
        """Initialize the mongodb service"""
        if not self._initialized:
            self.client = None

            # NOTE: Use that if you want to use a different collection name in your code and in your database (e.g. for versioning or different projects)
            self.collections_mapping = (
                collections_mapping
                if collections_mapping is not None
                else {
                    "results": "results",
                    "prompt_items": "prompt_items",
                    "primitive_viability": "primitive_viability",
                    "locks": "locks",
                    "runs": "runs",
                    "bet_generations": "bet_generations"
                }
            )
            self.collections = {}
            self.db_name = os.environ["MONGODB_DB_NAME"]
            self.self_hosted = self_hosted

    def test_connection(self):
        """Test the connection to the mongodb database"""
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            raise Exception("Failed to connect to MongoDB")

    def connect(self):
        """Connect to the mongodb database"""
        if self.client is None:
            connection_str = os.environ["MONGODB_URI"]
            self.client = MongoClient(
                connection_str, ssl=not self.self_hosted, tlsAllowInvalidCertificates=self.self_hosted
            )

    def close(self):
        """Close the connection to the mongodb database"""
        self.client.close()

    def initialize_collections(self):
        """Initialize the collections"""
        try:
            collection_names = self.client[self.db_name].list_collection_names()
            for collection_name, collection in self.collections_mapping.items():
                if collection not in collection_names:
                    self.client[self.db_name].create_collection(collection)
                self.collections[collection_name] = self.client[self.db_name][collection]
        except Exception as e:
            raise e

    def initialize(self):
        """Initialize the mongodb database"""
        try:
            logger.debug("Initializing mongodb database...")
            self.connect()
            self.test_connection()
            self.initialize_collections()
            self._initialized = True
            logger.debug("Database initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise e

    def get_collection(self, 
        collection_name: str
    ) -> Collection:
        """Get a collection by name"""
        return self.collections[collection_name]

    def get_collections(self) -> Dict[str, Collection]:
        """Get all collections"""
        return self.collections

    def get_client(self) -> MongoClient:
        """Get the client"""
        return self.client

# Create a global instance
database = MongodbService()
database.initialize()
