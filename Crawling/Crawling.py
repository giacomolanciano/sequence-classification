import unicodecsv as csv
from bs4 import BeautifulSoup as bs
import requests

class Crawler(object):
    def __init__(self,):
        self.pdb_dic = {}
        self.class_dic = {}

    def get_pdb_seq_dic(self,datapath):
        with open(datapath,'rb') as file:
            reader = csv.DictReader(file,delimiter = "\t", encoding="utf-8")
            for i,row in enumerate(reader):
                pdb_id = row['PDB']
                HTTP_request = requests.get('http://www.rcsb.org/pdb/files/fasta.txt?structureIdList=' + pdb_id)
                seq = HTTP_request.text
                self.parse_protein_seq(seq)
                self.pdb_dic[pdb_id] = seq
                HTTP_request = requests.get('http://www.rcsb.org/pdb/explore/explore.do?structureId=' + pdb_id)

                soup = bs(HTTP_request.text,"lxml")
                #ci serve solo il primo ul
                classification_label = soup.find('ul', attrs={'class': 'list-unstyled'}).find('li').find('a').get_text().strip()
                self.class_dic[classification_label] = 1
                print(i)
            print(len(self.class_dic))



    def parse_protein_seq(self,seq):
        seq =seq.split('>')[1].split('\n',maxsplit= 1)[1].replace("\n","")
        return seq


c = Crawler()
c.get_pdb_seq_dic("../Data/Table_PDB_Domain.tsv")
print(c.pdb_dic)