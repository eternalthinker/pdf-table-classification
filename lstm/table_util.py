import re
from bs4 import BeautifulSoup
import os
from gensim_load import log

def parse_table(table_content):
    soup = BeautifulSoup(table_content, 'html.parser')
    row_tags = soup.find_all('tr')
    rows = []
    for row_tag in row_tags:
        col_tags = row_tag.find_all('td')
        cols = []
        for col_tag in col_tags:
            colspan =  col_tag.get('colspan') or 0
            colspan = int(colspan)
            cell = str(col_tag.contents[0])
            # Remove tags
            cell = re.sub(r"<.*?>", r"", cell)
            cell = cell.replace(" .", "").strip()
            cell = cell.replace("$", "").strip()
            clean_cell = re.sub(r"[\(\),]", r"", cell)
            try:
                cell = float(clean_cell)
            except ValueError:
                pass # Not a float
            # print("Cell: {}, IsNumber: {}".format(cell, isinstance(cell, float)))
            cols.append((colspan, cell))
            # print(cols[-1])
        rows.append(cols)
        # print(len(cols))
    log(rows[0])
    log(rows[1])
    log("------ END OF TABLE ----")
    return rows
    


if __name__ == "__main__":
    f = os.path.join('data', '{}.html'.format("00980156_1"))
    with open(f, 'r', encoding='utf-8') as openf:
        s = openf.read()
        parse_table(s)
