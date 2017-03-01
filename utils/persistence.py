import sqlite3
from utils.constants import DATABASE


def insert_protein(pdb_id, sequence, class_label):
    """
    Insert protein data into db.
    :param pdb_id: protein PDB identifier.
    :param sequence: protein amino acids sequence.
    :param class_label: protein classification.
    """
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    try:
        cursor.execute('''INSERT INTO protein VALUES (?,?,?)''', (pdb_id, sequence, class_label))
    except sqlite3.IntegrityError as err:
        print(err)
    connection.commit()
    connection.close()


def is_known_protein(pdb_id):
    """
    Tells whether a protein is already in db or not.
    :param pdb_id: protein PDB identifier.
    :return: True if already in, False otherwise.
    """
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute('''SELECT * FROM protein WHERE pdb_id = ?''', (pdb_id,))
    for _ in cursor:
        connection.commit()
        connection.close()
        return True  # if result contains at least one tuple, return True
    connection.commit()
    connection.close()
    return False


def get_proteins_unique_labels():
    """
    Count how many distinct proteins labels are stored in db.
    :return: The number of distinct labels.
    """
    result = 0
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute('''SELECT COUNT(DISTINCT class_label) FROM protein''')
    for row in cursor:
        result = row[0]
    connection.commit()
    connection.close()
    return result


if __name__ == '__main__':

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS protein (pdb_id PRIMARY KEY, sequence, class_label)''')

    # Delete tables
    # c.execute("DROP TABLE protein")

    # Delete tables rows
    # c.execute("DELETE FROM protein")

    # Show tables
    # print('\nprotein')
    # c.execute("SELECT * FROM protein")
    # for row in c:
    #     print(row)

    # Show class labels items numbers
    # print('\nclass labels')
    # c.execute('''SELECT class_label, COUNT(*) items FROM protein GROUP BY class_label ORDER BY items DESC''')
    # for row in c:
    #     print(row)

    # Save (commit) the changes
    conn.commit()
    conn.close()
