import sqlite3
import os

DB_PATH = "data/progol.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print("Database not found. Initializing new database.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get current columns
    cursor.execute("PRAGMA table_info(matches)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    
    # Columns to add
    new_cols = [
        ('odds_home', 'FLOAT'),
        ('odds_draw', 'FLOAT'),
        ('odds_away', 'FLOAT'),
        ('odds_movement', 'FLOAT'),
        ('home_xg', 'FLOAT'),
        ('away_xg', 'FLOAT'),
        ('home_form', 'TEXT'),
        ('away_form', 'TEXT'),
        ('venue_id', 'INTEGER'),
        ('venue_surface', 'TEXT'),
        ('h2h_home_wins', 'INTEGER'),
        ('h2h_draws', 'INTEGER'),
        ('h2h_away_wins', 'INTEGER')
    ]
    
    print("Checking for missing columns...")
    added_count = 0
    for col_name, col_type in new_cols:
        if col_name not in existing_cols:
            print(f"Adding column: {col_name}")
            try:
                cursor.execute(f"ALTER TABLE matches ADD COLUMN {col_name} {col_type}")
                added_count += 1
            except Exception as e:
                print(f"Error adding {col_name}: {e}")
                
    conn.commit()
    conn.close()
    print(f"Migration complete. Added {added_count} columns.")

if __name__ == "__main__":
    migrate()
