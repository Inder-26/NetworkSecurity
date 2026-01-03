
import os
import sys
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient

# Load environment variables
load_dotenv()

mongo_url = os.getenv("MONGO_DB_URL")

print(f"Checking connection for URL: {mongo_url[:20]}..." if mongo_url else "No MONGO_DB_URL found in .env")

if not mongo_url:
    print("❌ MONGO_DB_URL is not set in .env")
    sys.exit(1)

try:
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    # Send a ping to confirm a successful connection
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB!")
    
    # List databases to be sure
    dbs = client.list_database_names()
    print(f"   Databases found: {dbs}")
    
except Exception as e:
    print(f"❌ Connection Failed: {e}")
