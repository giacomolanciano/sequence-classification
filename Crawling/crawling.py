import unicodecsv as csv
from bs4 import BeautifulSoup as bs
import requests

TABLE_PDB_FILE = '../data/Table_PDB_Domain.tsv'
FASTA_URL = 'http://www.rcsb.org/pdb/files/fasta.txt?structureIdList='
CLASS_URL = 'http://www.rcsb.org/pdb/explore/explore.do?structureId='
PDB_KEY = 'PDB'


class Crawler(object):
    def __init__(self, ):
        self.pdb_dic = {}
        self.class_dic = {}

    def get_pdb_seq_dic(self, data_path):
        with open(data_path, 'rb') as file:
            reader = csv.DictReader(file, delimiter='\t', encoding='utf-8')
            for i, row in enumerate(reader):
                pdb_id = row[PDB_KEY]
                http_request = requests.get(FASTA_URL + pdb_id)
                seq = http_request.text
                self.parse_protein_seq(seq)
                self.pdb_dic[pdb_id] = seq
                http_request = requests.get(CLASS_URL + pdb_id)

                soup = bs(http_request.text, 'lxml')

                classification_label = soup.find('ul', attrs={'class': 'list-unstyled'}).find('li').find('a') \
                    .get_text().strip()

                self.class_dic[classification_label] = 1
                print(i)
            print(len(self.class_dic))

    @staticmethod
    def parse_protein_seq(seq):
        seq = seq.split('>')[1].split('\n', maxsplit=1)[1].replace('\n', '')
        return seq


if __name__ == '__main__':
    c = Crawler()
    c.get_pdb_seq_dic(TABLE_PDB_FILE)
    print(c.pdb_dic)
