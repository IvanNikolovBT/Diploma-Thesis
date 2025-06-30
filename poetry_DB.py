import psycopg2
from typing import Optional
import re

class PoetryDB:
    def __init__(self) -> None:
        self.conn: psycopg2.extensions.connection = psycopg2.connect(
            host="localhost",
            database="poetry_db",
            user="ivanadmin",
            password="ivanadmin"
        )

    def get_author_id(self, full_name: str) -> Optional[int]:
        "Return the author’s id if found, otherwise None."
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM author WHERE full_name = %s;",
                (full_name,)
            )
            result = cur.fetchone()
        return result[0] if result else None
    
    def get_book_with_author_id(self, author_id, title):
        "Return the book_id of the book with the pair author_id and title"
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT book_id FROM book_table WHERE author_id = %s AND book_title = %s;",
                (author_id, title)
            )
            result = cur.fetchone()
        return result[0] if result else None

    def insert_author(self, full_name: str) -> int:
        "Insert a new author and return its generated id."
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
        "Ensure the author exists (insert if missing), then insert the song."
        try:
            author_id = self.get_author_id(author_full_name)
            if author_id is None:
                author_id = self.insert_author(author_full_name)
            self.insert_song(author_id, book_title, title, corpus)
        except Exception as e:
            print("Error inserting song with author:", e)

    def close(self) -> None:
        "Close the database connection."
        self.conn.close()
        
    def insert_book(self, author, book_title, date, page_link, scraped_from, object_label):
        "Insert the book and return its ID"
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.book_table (author_id, book_title, scraped_from, link, date, object_label)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING book_id;
                    """,
                    (author, book_title, scraped_from, page_link, date, object_label)
                )
                book_id: int = cur.fetchone()[0]
            self.conn.commit()
            print(f"Book '{book_title}' inserted with ID {book_id} for author {author}")
            return book_id
        except Exception as e:
            self.conn.rollback()
            print(f"[ERROR] Failed to insert book '{book_title}': {e}")
            return None

    def insert_biography(self,author_id,text,link):
        "Insert the biography for the author with ID"
        try:
            
            is_present=self.get_author_biography(author_id)
            if is_present is not None:
                print(f'Author {author_id} already has a biography')
                return None
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.biographies (author_id,text,link)
                    VALUES (%s, %s,%s)
                    RETURNING biography_id;
                    """,
                    (author_id, text,link)
                )
                biography_id: int = cur.fetchone()[0]
            self.conn.commit()
            print(f"Inserted biography for '{author_id}' from link {link}")
            return biography_id
        except Exception as e:
            self.conn.rollback()
            print(f"[ERROR] Failed to insert biography from link '{link}': {e}")
            return None
    def get_wiki_link_for_author(self, author_id: str) -> Optional[str]:
        "Return the author’s id if found, otherwise None."
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT link FROM biographies WHERE author_id = %s;",
                (author_id)
            )
            result = cur.fetchone()
        return result[0] if result else None
        
    def get_all_authors(self) -> Optional[list[str]]:
        "Returns a list of all of the authors in the database"
        with self.conn.cursor() as cur:
            cur.execute("SELECT full_name FROM author;")
            result = cur.fetchall()
        return [row[0] for row in result] if result else None

    def update_author(self, author_id, gender=None, place_of_birth=None, date_of_birth=None,
                  date_of_death=None, place_of_death=None, full_name=None):
        """
        Updates only the non-None fields for a given author.
        """
        fields = []
        values = []

        if gender is not None:
            fields.append("gender = %s")
            values.append(gender)
        if place_of_birth is not None:
            fields.append("place_of_birth = %s")
            values.append(place_of_birth)
        if date_of_birth is not None:
            fields.append("date_of_birth = %s")
            values.append(date_of_birth)
        if date_of_death is not None:
            fields.append("date_of_death = %s")
            values.append(date_of_death)
        if place_of_death is not None:
            fields.append("place_of_death = %s")
            values.append(place_of_death)
        if full_name is not None:
            fields.append("full_name = %s")
            values.append(full_name)

        if not fields:
            print(f"[INFO] No fields to update for author ID {author_id}.")
            return

        values.append(author_id)
        query = f"UPDATE author SET {', '.join(fields)} WHERE id = %s;"

        with self.conn.cursor() as cur:
            cur.execute(query, tuple(values))

        self.conn.commit()
        print(f"[INFO] Updated author ID {author_id} with fields: {', '.join(f.split('=')[0].strip() for f in fields)}.")
    def get_author_biography(self,author_id):
        "Returns the authors biography"
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT text FROM biographies WHERE author_id = {author_id};")
            result = cur.fetchone()
        return result[0] if result else None  
    def insert_scraping_info(self,text_file_location:str,scraped_from:str):
        "Add information regarding the book."
        
        with open(text_file_location) as f:
            
            content=f.read()
            page_links=re.findall("Page link: (.*)",content)
            book_titles=re.findall("Book: (.*)",content)
            object_labels=re.findall("Object Label: (.*)",content)
            dates=re.findall("Date: (.*)",content)        
            authors=re.findall("Author: (.*)",content)   
            
            
            for i in range(len(page_links)):
                author_id=self.get_author_id(authors[i])
                if author_id is None:
                    self.insert_author(authors[i])
                else:
                    print(f"1. Author {authors[i]} is already inserted. Going straight to book insertion")
                book_id=self.get_book_with_author_id(author_id,book_titles[i])
                if book_id is None:
                    self.insert_book(author_id,book_titles[i],dates[i],page_links[i],scraped_from,object_labels[i])
                else:
                    print(f"Book with title {book_titles[i]} and author {authors[i]} is already inserted.")
                
    def insert_kik_song(self, author_name: str, song_name: str, context: str, song_text: str):
        with self.conn.cursor() as cur:
            author_id = self.get_author_id(author_name)
            if author_id is None:
                author_id=self.insert_author(author_name)
            
            song_id=self.get_kik_song_id(song_name)
            if song_id is None:
                cur.execute("""
                    INSERT INTO song_kafe_kniga (author,author_id, context, song_title, song_text)
                    VALUES (%s,%s, %s, %s, %s)
                """, (author_name,author_id, context, song_name, song_text))
                self.conn.commit()
                print(f"Inserted '{song_name}' by {author_name}.")
            
            else:
                print(f'Song {song_name} already has id {song_id}')
                
        
    def get_kik_song_id(self,song_name:str):
        "Returns the kik song"
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM song_kafe_kniga WHERE song_title = %s", (song_name,))
            result = cur.fetchone()
        return result[0] if result else None  

            
                

            