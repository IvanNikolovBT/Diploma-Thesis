import os
import re
import pandas as pd

base_dir = '/home/ivan/Desktop/Diplomska/Aditional_authors'

records = []  # collect all parsed blocks here

for entry in os.listdir(base_dir):
    path = os.path.join(base_dir, entry)
    
    if os.path.isdir(path):  
        for sub in os.listdir(path):
            file_path = os.path.join(path, sub)
            
            if os.path.isfile(file_path):  
                print(f"Reading file: {sub}")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                blocks = re.split(r"=+\n", content.strip())

                current_source = None

                current_author = None
                context_value = None  

                for i, block in enumerate(blocks):
                    record = {}

                    
                    if i == 0:
                        source_match = re.search(r"Source:\s*(.*)", block)
                        author_match = re.search(r"Author:\s*(.*)", block)
                        context_match = re.search(r"Context:\s*(.*?)(?=\n=+)", block, re.S)

                        if source_match:
                            current_source = source_match.group(1).strip()
                        if author_match:
                            current_author = author_match.group(1).strip()
                        if context_match:
                            context_value = context_match.group(1).strip()

                    record["Source"] = current_source
                    
                    record["Author"] = current_author
                    record["Context"] = context_value  
                    add_ctx_match = re.search(
                        r"Additional_Context:\s*(.*?)(?=Song title:|Song:|$)", block, re.S
                    )
                    record["Additional_Context"] = add_ctx_match.group(1).strip() if add_ctx_match else None

                    song_title_match = re.search(r"Song title:\s*(.*)$", block, re.MULTILINE)
                    if song_title_match and song_title_match.group(1).strip():
                        record["Song title"] = song_title_match.group(1).strip()
                    else:
                        record["Song title"] = "FIX"


                    song_match = re.search(r"Song:\s*(.*)", block, re.S)
                    record["Song"] = song_match.group(1).strip() if song_match else None

                    
                    if record["Song"]:
                        records.append(record)

df = pd.DataFrame(records)
df.to_csv("output.csv", index=False, encoding="utf-8")
print("✅ Saved everything into output.csv")
