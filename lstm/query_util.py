OP_LT = 'lt'
OP_GT = 'gt'
OP_AND = 'and'

def parse_query(query):
    if OP_AND in query:
        return parse_and(query)
    else:
        return [parse_cond(query)]

def parse_and(query):
    components = query.split(OP_AND)
    components = list(map(lambda s: s.strip(), components))
    conds = list(map(parse_cond, components))
    return conds

def parse_cond(query):
    op, num = query.split()
    num = float(num)
    if op == OP_GT:
        return lambda n: n[1] > num if isinstance(n[1], float) else False
    elif op == OP_LT:
        return lambda n: n[1] < num if isinstance(n[1], float) else False
