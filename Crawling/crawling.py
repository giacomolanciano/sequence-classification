import os

import unicodecsv as csv
import requests
from bs4 import BeautifulSoup as bs

import utils.persistence as persistence
from utils.constants import DATA_FOLDER

TABLE_PDB_FILE = os.path.join(DATA_FOLDER, 'Table_PDB_Domain.tsv')
PDB_FILE = os.path.join(DATA_FOLDER, 'pdb_seqres.txt')
FASTA_URL = 'http://www.rcsb.org/pdb/files/fasta.txt?structureIdList='
CLASS_URL = 'http://www.rcsb.org/pdb/explore/explore.do?structureId='
PDB_ID_HEADER = 'PDB'


class Crawler(object):
    def __init__(self, progress=True):
        self.progress = progress
        self.errored_pdb_id = []

    def build_proteins_dataset(self, data_path):
        with open(data_path, 'rb') as file:
            reader = csv.DictReader(file, delimiter='\t', encoding='utf-8')
            for i, row in enumerate(reader):
                if self.progress:
                    print(i)

                pdb_id = row[PDB_ID_HEADER]

                # check whether protein is already in db to avoid costly network operations
                if not persistence.is_known_protein(pdb_id):
                    try:

                        # download sequence in FASTA format
                        fasta_http_request = requests.get(FASTA_URL + pdb_id)
                        sequence = self._parse_protein_sequence(fasta_http_request.text)

                        # download class label corresponding to sequence
                        class_http_request = requests.get(CLASS_URL + pdb_id)
                        soup = bs(class_http_request.text, 'lxml')
                        class_label = soup.find('ul', attrs={'class': 'list-unstyled'}).find('li').find('a')\
                            .get_text().strip()

                        # insert protein data in db
                        persistence.insert_protein(pdb_id, sequence, class_label)

                    except (TimeoutError, ConnectionError, AttributeError) as err:
                        # if network error occurs, skip
                        if self.progress:
                            print(err)
                        self.errored_pdb_id.append(pdb_id)
                        continue

    @staticmethod
    def _parse_protein_sequence(fasta_sequence):
        return fasta_sequence.split('>')[1].split('\n', maxsplit=1)[1].replace('\n', '')

    def build_proteins_dataset_local(self, data_path):
        with open(data_path, 'r') as file:

            i = 0
            header_line = file.readline()
            sequence_line = file.readline()

            while header_line is not '':
                pdb_id = Crawler._parse_protein_pdb_id(header_line)

                if self.progress:
                    print(i)
                    print(pdb_id)

                # check whether protein is already in db to avoid costly network operations
                if not persistence.is_known_protein(pdb_id):
                    try:

                        # download class label corresponding to sequence
                        class_http_request = requests.get(CLASS_URL + pdb_id)
                        soup = bs(class_http_request.text, 'lxml')
                        class_label = soup.find('ul', attrs={'class': 'list-unstyled'}).find('li').find('a')\
                            .get_text().strip()

                        # insert protein data in db
                        persistence.insert_protein(pdb_id, sequence_line, class_label)

                    except (TimeoutError, ConnectionError) as err:
                        # if network error occurs, skip
                        if self.progress:
                            print(err)
                        self.errored_pdb_id.append(pdb_id)
                        continue

                i += 1
                header_line = file.readline()
                sequence_line = file.readline()

    @staticmethod
    def _parse_protein_pdb_id(header):
        return str(header[1:5]).upper()

if __name__ == '__main__':
    c = Crawler()

    # c.build_proteins_dataset(TABLE_PDB_FILE)

    c.build_proteins_dataset_local(PDB_FILE)
    if c.errored_pdb_id:
        print('errored pdb_ids: ' + str(c.errored_pdb_id))
    print('unique labels: ' + str(persistence.get_proteins_unique_labels()))
