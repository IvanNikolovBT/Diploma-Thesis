import chromadb

client = chromadb.PersistentClient(
                path="vector_db",
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
print(client.list_collections())
