def resolve_compounds(token):
    compound = []
    for dep, child_list in token.children().items():
        for child in child_list:
            if child.dep() == 'compound':
                compound = resolve_compounds(child)
    compound.append(token)
    return compound


def resolve_auxiliaries(token):
    auxiliary = []
    for dep, child_list in token.children().items():
        for child in child_list:
            if child.dep() == 'auxiliary':
                auxiliary = resolve_auxiliaries(child)
    auxiliary.append(token)
    return auxiliary
