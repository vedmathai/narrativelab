def resolve_compounds(token):
    compound = []
    for dep, child_list in token.children().items():
        for child in child_list:
            if child.dep() == 'compound':
                compound = resolve_compounds(child)
    compound.append(token)
    return compound

def resolve_auxiliaries(token):
    auxiliaries = []
    for dep, child_list in token.children().items():
        for child in child_list:
            if child.dep() in ['auxiliary', 'prt']:
                auxiliaries = resolve_auxiliaries(child)
    auxiliaries.extend([token])
    auxiliaries = sorted(auxiliaries, key=lambda x: x.i())
    return auxiliaries

def get_all_children_tokens(token):
    queue = [token]
    tokens = []
    while len(queue) > 0:
        token = queue.pop(0)
        for children in token.children().values():
            queue.extend(children)
        tokens.append(token)
    return tokens

def resolve_absolute_time_compounds(token):
    compound = []
    for dep, child_list in token.children().items():
        for child in child_list:
            if child.dep() in ['nummod', 'compound'] and child.entity_type() == 'DATE':
                compound = resolve_compounds(child)
    compound.append(token)
    return compound