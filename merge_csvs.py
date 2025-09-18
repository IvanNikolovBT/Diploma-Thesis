import pandas as pd

file1 = "/home/ivan/Desktop/Diplomska/classification/song_kafe_kniga.csv"
file2 = "/home/ivan/Desktop/Diplomska/output.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

df1 = df1.rename(columns={
    "author": "author",
    "context": "context",
    "song_title": "song_title",
    "song_text": "song_text",
    "author_id": "author_id"
})
df1["additional_context"] = ""
df1["source"] = "song_kafe_kniga"

df2 = df2.rename(columns={
    "Author": "author",
    "Context": "context",
    "Additional_Context": "additional_context",
    "Song title": "song_title",
    "Song": "song_text",
    "Source": "source"
})
df2["author_id"] = ""

df2.insert(0, "id", range(df1["id"].max() + 1, df1["id"].max() + 1 + len(df2)))

final_columns = [
    "id", "author", "context", "additional_context",
    "song_title", "song_text", "author_id", "source"
]

merged = pd.concat([df1[final_columns], df2[final_columns]], ignore_index=True)

output_file = "/home/ivan/Desktop/Diplomska/merged.csv"
merged.to_csv(output_file, index=False)

print(f"Merged CSV saved to {output_file}")
