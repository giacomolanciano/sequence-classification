""" Module containing all constants. """
import os
import sqlite3 as sql

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
DATABASE = os.path.join(DATA_FOLDER, 'proteins.db')


if __name__ == '__main__':
    print(DATA_FOLDER)
    print(DATABASE)
    con = sql.connect(DATABASE)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    available_table = (cur.fetchall())
    print(available_table)
    curr =  cur.execute("SELECT * FROM protein;")
    dict = {}

    for row in cur:
        try:
            dict[row[2]] = dict[row[2]]+ 1
        except KeyError:
            dict[row[2]] = 1
    val = 0
    for k,v in dict.items():
        print(k,v)
        val = val + v
    print(val)