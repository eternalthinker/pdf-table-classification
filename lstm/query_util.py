OP_LT = 'lt'
OP_GT = 'gt'
OP_AND = 'and'
OP_YEAR = 'year'


def parse_query(query):
    if OP_AND in query:
        return parse_and(query)
    elif OP_YEAR in query:
        return {
            'col': [parse_year(query)],
            'row': []
        }
    else:
        return {
            'col': [],
            'row': [parse_cond(query)]
        }

def parse_and(query):
    components = query.split(OP_AND)
    components = list(map(lambda s: s.strip(), components))
    row_components = list(filter(lambda q: OP_YEAR not in q, components))
    col_components = list(filter(lambda q: OP_YEAR in q, components))
    row_conds = list(map(parse_cond, row_components))
    col_conds = list(map(parse_year, col_components))
    return {
        'row': row_conds,
        'col': col_conds
    }

def parse_year(query):
    op, num = query.split()
    num = float(num)
    if op == OP_YEAR:
        return lambda n: n[0] == num

def parse_cond(query):
    op, num = query.split()
    num = float(num)
    if op == OP_GT:
        return lambda n: n[1] > num if isinstance(n[1], float) else False
    elif op == OP_LT:
        return lambda n: n[1] < num if isinstance(n[1], float) else False
