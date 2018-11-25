#!/usr/bin/python
import psycopg2
import table_util
import os

 
def fill_table(fname):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = {
            'host': 'localhost',
            'database': 'test',
            'user': 'postgres',
            'password': 'password123'
        }
 
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
 
        # create a cursor
        cur = conn.cursor()

        table_name = 't_{}'.format(fname)

        drop_sql = '''DROP TABLE IF EXISTS {}'''.format(table_name)
        create_table_sql = '''CREATE TABLE {} (
            row_header TEXT
        )'''.format(table_name)

        insert_row_sql = '''INSERT INTO {} (row_header) VALUES ('{}')'''

        cur.execute(drop_sql)
        cur.execute(create_table_sql)
        f = os.path.join('data', '{}.html'.format(fname))
        with open(f, 'r', encoding='utf-8') as openf:
            s = openf.read()
            parsed_table = table_util.parse_table(s)
            rows = parsed_table['rows']
            for row in rows:
                row_header = row[0][1]
                query = insert_row_sql.format(table_name, row_header)
                cur.execute(query)
        
        conn.commit()

        
        # execute a statement
        #print('Similarity threshold:')
        #cur.execute('SELECT show_limit()')
 
        # display the PostgreSQL database server version
        #db_version = cur.fetchone()
        #print(db_version)
       
        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')
 
def compare_tables(fname1, fname2, fileid):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = {
            'host': 'localhost',
            'database': 'test',
            'user': 'postgres',
            'password': 'password123'
        }
 
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
 
        # create a cursor
        cur = conn.cursor()

        t1 = 't_{}'.format(fname1)
        t2 = 't_{}'.format(fname2)

        compare_sql = '''
            SELECT
            * 
            FROM (
            SELECT
                ROW_NUMBER() OVER (PARTITION BY row_header ORDER BY sim DESC) AS r,
                t.*
            FROM
                (
                    SELECT t1.row_header, t2.row_header as sim_row, similarity(t1.row_header, t2.row_header) AS sim
                    FROM   {} t1
                    LEFT OUTER JOIN   {} t2 ON t1.row_header % t2.row_header
                ) t) x
            WHERE
            x.r <= 1;
        '''

        cur.execute(compare_sql.format(t1, t2))
        f = os.path.join('output', 'comp{}.csv'.format(fileid))
        with open(f, 'w', encoding='utf-8') as openf:
            for row in cur:
                row = [str(item) for item in row]
                openf.write(",".join(row) + "\n")
        
        conn.commit()

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')
 
if __name__ == '__main__':
    tables = [
        '01443569_2', '01777354_2',
       "01218676_4", '01552305_4',
       "01101541_4", '01221436_4',
       "00996787_2", '01820054_2',
       "01335759_3", '01781810_3',
       "01552305_1", '01660919_1',
       "01445667_1", '01781810_1',
       "00992279_2", '01556590_2',
       "01333323_4", '01552305_3',
       "01772045_5", '01888225_4'
    ]
    
    #for table in tables:
    #    fill_table(table)

    for i in range(0, len(tables)//2):
        compare_tables(tables[i*2], tables[i*2 + 1], i+1)
