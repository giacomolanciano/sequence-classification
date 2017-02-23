import sqlite3
from utils.constants import DATABASE


def insert_protein(pdb_id, sequence, class_label):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    try:
        cursor.execute('''INSERT INTO protein VALUES (?,?,?)''', (pdb_id, sequence, class_label))
    except sqlite3.IntegrityError as err:
        print(err)
    connection.commit()
    connection.close()


if __name__ == '__main__':

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS protein (pdb_id PRIMARY KEY, sequence, class_label)''')

    # Delete tables
    # c.execute("DROP TABLE protein")

    # Delete tables rows
    # c.execute("DELETE FROM protein")

    # Show  tables
    print('\nprotein')
    c.execute("SELECT * FROM protein")
    for row in c:
        print(row)

    # Save (commit) the changes
    conn.commit()
    conn.close()
