import psycopg2
from typing import Optional


class PoetryDB:
    def __init__(self) -> None:
        self.conn: psycopg2.extensions.connection = psycopg2.connect(
            host="localhost",
            database="poetry_db",
            user="ivanadmin",
            password="ivanadmin"
        )

    def get_author_id(self, full_name: str) -> Optional[int]:
        """Return the author’s id if found, otherwise None."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM author WHERE full_name = %s;",
                (full_name,)
            )
            result = cur.fetchone()
        return result[0] if result else None

    def insert_author(self, full_name: str) -> int:
        """Insert a new author and return its generated id."""
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO author (full_name) VALUES (%s) RETURNING id;",
                (full_name,)
            )
            author_id: int = cur.fetchone()[0]
        self.conn.commit()
        print(f"Author '{full_name}' inserted with ID {author_id}.")
        return author_id

    def insert_song(self,
                    author_id: int,
                    book_title: str,
                    title: str,
                    corpus: str
                    ) -> None:
        """Insert a song linked to an existing author."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO song (author_id, book_title, title, corpus)
                VALUES (%s, %s, %s, %s);
                """,
                (author_id, book_title, title, corpus)
            )
        self.conn.commit()
        print("Song inserted successfully!")

    def insert_song_with_author(self,
                                book_title: str,
                                title: str,
                                corpus: str,
                                author_full_name: str
                                ) -> None:
        """
        Ensure the author exists (insert if missing), then insert the song.
        """
        try:
            author_id = self.get_author_id(author_full_name)
            if author_id is None:
                author_id = self.insert_author(author_full_name)
            self.insert_song(author_id, book_title, title, corpus)
        except Exception as e:
            print("Error inserting song with author:", e)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        
    def insert_scraping_info(self,text_file_location:str):
        pass 

