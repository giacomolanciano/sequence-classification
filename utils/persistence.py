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
        cursor.execute('''INSERT INTO protein_dupl VALUES (?,?,?)''', (pdb_id, sequence, class_label))
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
    cursor.execute('''SELECT * FROM protein_dupl WHERE pdb_id = ?''', (pdb_id,))
    for _ in cursor:
        connection.commit()
        connection.close()
        return True  # if result contains at least one tuple, return True
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
    connection.close()
    return result


def get_data_by_table(table_name="protein"):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM ' + table_name)
    table = cursor.fetchall()
    connection.close()
    return table


def get_data_by_label(label_name, table_name="protein"):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM ' + table_name + ' WHERE class_label =  ?', (label_name,))
    table = cursor.fetchall()
    connection.close()
    return table


def get_sequence_label_data_by_label(label_name, table_name="protein"):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()
    cursor.execute('SELECT  sequence, class_label FROM ' + table_name + ' WHERE class_label =  ?', (label_name,))
    table = cursor.fetchall()
    connection.close()
    return table


def _filter_in_label_duplicates():
    """
    Pass data into new table to filter in-label duplicates.
    """
    connection = sqlite3.connect(DATABASE)
    cursor1 = connection.cursor()
    cursor1.execute('''SELECT * FROM protein_dupl''')
    for row in cursor1:
        cursor2 = connection.cursor()
        try:
            cursor2.execute('''INSERT INTO protein VALUES (?,?,?)''', (row[0], row[1], row[2]))
        except sqlite3.IntegrityError as err:
            print(err)
    connection.commit()
    connection.close()


if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # Create tables
    c.execute(
        '''CREATE TABLE IF NOT EXISTS protein (pdb_id, sequence, class_label, PRIMARY KEY(sequence, class_label))''')
    c.execute('''CREATE TABLE IF NOT EXISTS protein_dupl (pdb_id PRIMARY KEY, sequence, class_label)''')

    # Show tables
    # c.execute("SELECT * FROM protein")
    # for r in c:
    #     print(r)

    # print(get_data_by_label('HYDROLASE'))
    # print(get_sequence_label_data_by_label('HYDROLASE'))

    # Save (commit) the changes
    conn.commit()
    conn.close()
