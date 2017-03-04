import sqlite3 as sql
from utils.constants import DATABASE

class Domain(object):
    def __init__(self):
        self.connection = sql.connect(DATABASE)
        self.cursor = self.connection.cursor()

    def get_data_by_table(self,table_name = "protein"):
        self.cursor.execute("SELECT * FROM " + table_name)
        table = self.cursor.fetchall()
        my_table = []
        for row in table:
            record = [i.replace("\n","") for i in row]
            my_table.append(record)
        return my_table

    def get_data_by_label(self,label_name,table_name = "protein"):

        self.cursor.execute("SELECT * FROM " + table_name + " WHERE class_label =  ?", (label_name,))
        table = self.cursor.fetchall()
        my_table = []
        for row in table:
            record = [i.replace("\n","") for i in row]
            my_table.append(record)
        return my_table

    def get_sequance_label_data_by_label(self,label_name,table_name = "protein"):

        self.cursor.execute("SELECT  sequence, class_label FROM " + table_name + " WHERE class_label =  ?", (label_name,))
        table = self.cursor.fetchall()
        my_table = []
        for row in table:
            record = [i.replace("\n","") for i in row]
            my_table.append(record)
        return my_table


if __name__ == '__main__':
    my_domain = Domain()
    my_domain.get_data_by_table()
    my_domain.get_data_by_label("HYDROLASE")