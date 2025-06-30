import tkinter as tk
from tkinter import messagebox
from poetry_DB import PoetryDB


class SongForm(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Insert Song Data")
        self.db=PoetryDB()
        
        tk.Label(self, text="Author (max 50 chars):").grid(row=0, column=0, sticky="w")
        self.author_entry = tk.Entry(self, width=50)
        self.author_entry.grid(row=0, column=1)

        tk.Label(self, text="Context:").grid(row=1, column=0, sticky="w")
        self.context_text = tk.Text(self, height=4, width=38)
        self.context_text.grid(row=1, column=1)

        tk.Label(self, text="Song Title (max 50 chars):").grid(row=2, column=0, sticky="w")
        self.song_title_entry = tk.Entry(self, width=50)
        self.song_title_entry.grid(row=2, column=1)

        tk.Label(self, text="Song Text:").grid(row=3, column=0, sticky="w")
        self.song_text_text = tk.Text(self, height=6, width=38)
        self.song_text_text.grid(row=3, column=1)

        tk.Label(self, text="Author ID (integer, optional):").grid(row=4, column=0, sticky="w")
        self.author_id_entry = tk.Entry(self, width=50)
        self.author_id_entry.grid(row=4, column=1)

        submit_btn = tk.Button(self, text="Insert Song", command=self.submit)
        submit_btn.grid(row=5, column=0, columnspan=2, pady=10)

    def submit(self):
        author = self.author_entry.get().strip()
        context = self.context_text.get("1.0", tk.END).strip()
        song_title = self.song_title_entry.get().strip()
        song_text = self.song_text_text.get("1.0", tk.END).strip()
        author_id_str = self.author_id_entry.get().strip()

        if len(author) > 50:
            messagebox.showwarning("Warning", "Author name too long, trimming to 50 characters.")
            author = author[:50]

        if len(song_title) > 50:
            messagebox.showwarning("Warning", "Song title too long, trimming to 50 characters.")
            song_title = song_title[:50]

        author_id = None
        if author_id_str:
            if not author_id_str.isdigit():
                messagebox.showerror("Error", "Author ID must be an integer.")
                return
            author_id = int(author_id_str)

        data = {
            "author": author,
            "context": context,
            "song_title": song_title,
            "song_text": song_text,
            "author_id": author_id,
        }

        self.db.insert_kik_song(author_name=author,context=context,song_name=song_title,song_text=song_text)

if __name__ == "__main__":
    app = SongForm()
    app.mainloop()
