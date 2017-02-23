import unicodecsv as csv
from bs4 import BeautifulSoup as bs
import requests
import utils.persistence as persistence

TABLE_PDB_FILE = '../Data/Table_PDB_Domain.tsv'
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

                    except (TimeoutError, ConnectionError) as err:
                        # if network error occurs, skip
                        if self.progress:
                            print(err)
                        self.errored_pdb_id.append(pdb_id)
                        continue

    @staticmethod
    def _parse_protein_sequence(fasta_sequence):
        return fasta_sequence.split('>')[1].split('\n', maxsplit=1)[1].replace('\n', '')


if __name__ == '__main__':
    c = Crawler()
    c.build_proteins_dataset(TABLE_PDB_FILE)
    print('errored pdb_ids: ' + str(c.errored_pdb_id))
    print('# unique labels: ' + str(persistence.get_proteins_unique_labels()))
