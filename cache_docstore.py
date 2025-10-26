import pickle
from llama_index.core import StorageContext

# Path to your persistence directory (from your .env)
persist_dir = "local_data/private_gpt"

print("🐢 Loading StorageContext from JSON (this will be slow, just once)...")
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

# Save both docstore and index_store
print("💾 Caching docstore and index_store...")
with open(f"{persist_dir}/docstore_cache.pkl", "wb") as f:
    pickle.dump(storage_context.docstore, f)

with open(f"{persist_dir}/index_store_cache.pkl", "wb") as f:
    pickle.dump(storage_context.index_store, f)

print("✅ Done! Both docstore and index_store cached successfully.")
