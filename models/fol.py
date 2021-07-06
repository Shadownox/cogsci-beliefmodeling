import itertools

import numpy as np
import ccobra


# the basic objects are defined here for convenience and better readability
A = "A"
B = "B"
C = "C"
all_terms = [A, B, C]


def int_to_bools(num, digits):
    """ Converts an integer to an array of bools of its binary representation.

    Parameters
    ----------
    num : number
        The number to encode.

    digits : number
        The number of digits.

    Returns
    -------
    list(bool)
        Array of bools representing the number.

    """
    return [bool(num & (1<<n)) for n in range(digits)]

def get_all_expanded_sets(initial, subj, obj, terms):
    """ Extends sets by adding all combinations of the missing terms
    (e.g., the statement [A,B] becomes [[A,B], [A,B,C]].

    Parameters
    ----------
    initial : list(str)
        The statement to be expanded.

    subj : str
        The subject of the statement.

    obj : str
        The object of the statement.

    terms : list(str)
        The list of terms

    Returns
    -------
    list(list(str))
        The list containing all expansions possible based on the initial statement.

    """
    results = []

    # get all remaining terms (for normal syllogisms, this should be just one)
    remaining_terms = [x for x in terms if x != subj and x != obj]
    remaining_terms = np.array(remaining_terms)

    n_terms = len(remaining_terms)

    # add all combinations of the remaining terms to the initial statement
    for i in range(2**n_terms):
        mask = int_to_bools(i, n_terms)
        to_append = remaining_terms[mask]
        results.append(sorted(initial + to_append.tolist()))
    return results

def get_premise_meanings(quant, subj, obj):
    """ Translates the meaning of a premise into a list of positive statements
    (that must be fulfilled) and negative statements (which must be falsified).

    Parameters
    ----------
    quant : str
        The quantifier (A = All, E = No, I = Some, O = Some not)

    subj : str
        The subject of the premise.

    obj : str
        The object of the premise.

    Returns
    -------
    tuple(list(list(list(str))), list(list(str)))
        Tuple containing positive and negative statements.

    """
    positives = []
    negatives = []

    if quant == "A":
        # all subj are obj
        # There must be evidence for [subj and obj]
        positives.append(get_all_expanded_sets([subj, obj], subj, obj, all_terms))

        # Evidence for subj without obj must not exist
        negatives.extend(get_all_expanded_sets([subj], subj, obj, all_terms))
    elif quant == "E":
        # no subj are obj
        # There must be evidence for both, subj and obj, alone
        positives.append(get_all_expanded_sets([subj], subj, obj, all_terms))
        positives.append(get_all_expanded_sets([obj], subj, obj, all_terms))

        # Evidence for subj with obj must not exist
        negatives.extend(get_all_expanded_sets([subj, obj], subj, obj, all_terms))
        pass
    elif quant == "I":
        # some subj are obj
        # There must be evidence for [subj and obj]
        positives.append(get_all_expanded_sets([subj, obj], subj, obj, all_terms))
    elif quant == "O":
        # some subj are not obj
        # There must be evidence for subj without obj
        positives.append(get_all_expanded_sets([subj], subj, obj, all_terms))

    return (positives, negatives)

def all_combinations(worlds):
    """ For a list of worlds (worlds = a possible situation), calculate the combinations
    of them (each element from one world with the elements of the other worlds).
    Additionally, statements are converted to sets of tuples.

    Parameters
    ----------
    worlds : list(list(list(str)))
        list of worlds

    Returns
    -------
    list(set(tuple(str)))
        List containing all combinations of the initial world list.

    """
    tupled_worlds = []
    for world in worlds:
        tupled_worlds.append([tuple(x) for x in world])

    combinations = [set(c) for c in itertools.product(*tupled_worlds)]
    return combinations


def create_world_set(syl):
    """ Creates the list of worlds for a given syllogism.
    Each world represents a possible state given the premises.
    The worlds can then be used to check the conclusions against them.

    Parameters
    ----------
    syl : str
        Encoded syllogism (e.g., AA1), according to the encoding used in CCOBRA.

    Returns
    -------
    list(set(tuple(str)))
        List containing all worlds for the given syllogism.

    """
    quant1 = syl[0]
    quant2 = syl[1]
    figure = int(syl[2])

    subj1 = None
    subj2 = None
    obj1 = None
    obj2 = None

    # decode figures
    if figure == 1:
        subj1 = A
        obj1 = B
        subj2 = B
        obj2 = C
    elif figure == 2:
        subj1 = B
        obj1 = A
        subj2 = C
        obj2 = B
    elif figure == 3:
        subj1 = A
        obj1 = B
        subj2 = C
        obj2 = B
    elif figure == 4:
        subj1 = B
        obj1 = A
        subj2 = B
        obj2 = C

    # translate the premise meaning to positive and negative statements
    # These describe the areas of a venn diagram for which direct information
    # is given.
    p1_pos, p1_neg = get_premise_meanings(quant1, subj1, obj1)
    p2_pos, p2_neg = get_premise_meanings(quant2, subj2, obj2)

    positives = p1_pos + p2_pos
    negatives = set([tuple(x) for x in p1_neg + p2_neg])

    # calculate the statements that are missing
    # corresponds to areas in the venn diagram where no direct information is given
    all_parts_tuples = set()
    for pos in positives:
        for p_sub in pos:
            all_parts_tuples.add(tuple(p_sub))
    for neg in negatives:
        all_parts_tuples.add(neg)

    terms_array = np.array(all_terms)

    additions = []
    for i in range(1, 2**len(all_terms)):
        mask = int_to_bools(i, len(all_terms))
        other = tuple(terms_array[mask])
        if other not in all_parts_tuples:
            additions.append(",".join(other))

    additions = np.array(additions)

    # Clean the worlds by removing negatives
    cleaned_worlds = []
    for world in positives:
        cleaned_world = [x for x in world if tuple(x) not in negatives]
        cleaned_worlds.append(cleaned_world)

    # build the resulting states by building the combinations between the statements
    # This corresponds to integrating the premises
    final_worlds = all_combinations(cleaned_worlds)

    # Add the missing information (additions) calulcated above to each world.
    # Thereby, all combinations need to be used, as each addition represent a possible state.
    result = []
    for world in final_worlds:
        for i in range(2**len(additions)):
            mask = int_to_bools(i, len(additions))
            to_add = additions[mask]

            world_copy = world.copy()
            for a in to_add:
                a = tuple(a.split(","))
                world_copy.add(a)
            result.append(world_copy)

    return result

def check_conclusion_in_world(conclusion, world):
    """ Checks if a conclusion is in line with a single world.

    Parameters
    ----------
    conclusion : str
        Encoded conclusion (e.g., Aac), according to the encoding used in CCOBRA.
        NVC is not considered a possible conclusion at this point!

    world : set(tuple(str))
        A set of tuples describing a possible state according to the premises.

    Returns
    -------
    bool
        True, if the conclusion is in line with the world. False otherwise.

    """
    quant = conclusion[0]
    subj = None
    obj = None

    if conclusion.endswith("ac"):
        subj = A
        obj = C
    elif conclusion.endswith("ca"):
        subj = C
        obj = A

    if quant == "A":
        # if subj, then obj
        contains_subj = [x for x in world if subj in x]
        contains_obj = [obj in x for x in contains_subj]
        if not contains_obj:
            return False
        return np.all(contains_obj)
    elif quant == "E":
        # if subj, then not obj
        contains_subj = [x for x in world if subj in x]
        contains_not_obj = [obj not in x for x in contains_subj]
        if not contains_not_obj:
            return False
        return np.all(contains_not_obj)
    elif quant == "I":
        # if subj, then obj has to occur
        contains_subj = [x for x in world if subj in x]
        contains_obj = [obj in x for x in contains_subj]
        if not contains_obj:
            return False
        return np.any(contains_obj)
    elif quant == "O":
        # if subj, not obj has to occur
        contains_subj = [x for x in world if subj in x]
        contains_not_obj = [obj not in x for x in contains_subj]
        return np.any(contains_not_obj)


def check_conclusion(conclusion, worlds):
    """ Checks if a conclusion is in line all worlds. Note that NVC is not
    considered to be a valid conclusion at this point. Rather, if no other
    conclusion follows, NVC would be concluded.

    Parameters
    ----------
    conclusion : str
        Encoded conclusion (e.g., Aac), according to the encoding used in CCOBRA.
        NVC is not considered a possible conclusion at this point!

    worlds : list(set(tuple(str)))
        A list of worlds describing possible states according to the premises.

    Returns
    -------
    tuple(bool, bool)
        A tuple containing two bools. The first bool is true if the conclusion
        is possible in the given worlds, the second is true iff the conclusion
        neccessarily follows.

    """
    follows = True
    possible = False
    for world in worlds:
        holds = check_conclusion_in_world(conclusion, world)
        if holds:
            possible = True
        else:
            follows = False
    return (possible, follows)


def evaluate_conclusion(conclusion, syl):
    """ Checks if a conclusion neccessarily follows from a syllogism or if the
    conclusion is possible given the premises of the syllogism. Note that NVC is
    not considered to be a valid conclusion at this point. Rather, if no other
    conclusion follows, NVC would be concluded.

    Parameters
    ----------
    conclusion : str
        Encoded conclusion (e.g., Aac), according to the encoding used in CCOBRA.
        NVC is not considered a possible conclusion at this point!

    syl : str
        Encoded syllogism (e.g., AA1), according to the encoding used in CCOBRA.

    Returns
    -------
    tuple(bool, bool)
        A tuple containing two bools. The first bool is true if the conclusion
        is possible given the syllogism, the second is true iff the conclusion
        neccessarily follows.

    """
    worlds_for_syllog = create_world_set(syl)
    possible, follows = check_conclusion(conclusion, worlds_for_syllog)
    return possible, follows


def get_conclusions_for_syllog(syl):
    """ For each conclusion candidate and a given syllogism, it is calculated
    if the conclusion candidates are possible or follow from the syllogism.

    Parameters
    ----------
    syl : str
        Encoded syllogism (e.g., AA1), according to the encoding used in CCOBRA.

    Returns
    -------
    dict(str, tuple(bool, bool))
        A dictionary mapping conclusions to a tuple containing two bools. The
        first bool is true if the conclusion is possible given the syllogism,
        the second is true iff the conclusion neccessarily follows.

    """
    worlds_for_syllog = create_world_set(syl)

    conclusions_dict = {}
    # check conclusions
    for conclusion in ccobra.syllogistic.RESPONSES:
        if conclusion == "NVC":
            continue

        possible, follows = check_conclusion(conclusion, worlds_for_syllog)

        conclusions_dict[conclusion] = (possible, follows)
    return conclusions_dict


def get_valid_responses(syl):
    """ Calculates the valid responses for the given syllogism. If there is no
    valid response, NVC is concluded.

    Parameters
    ----------
    syl : str
        Encoded syllogism (e.g., AA1), according to the encoding used in CCOBRA.

    Returns
    -------
    list(str)
        List of valid responses.

    """
    concls = get_conclusions_for_syllog(syl)
    valids = [x for x, y in concls.items() if y[1]]
    if not valids:
        return ["NVC"]
    return valids


class FOL():
    def __init__(self):
        self.cache = {}
        pass

    def fit(self, train_data):
        pass

    def evaluate_conclusion(self, conclusion, syllogism):
        if (conclusion, syllogism) not in self.cache:
            possible, follows = evaluate_conclusion(conclusion, syllogism)
            self.cache[(conclusion, syllogism)] = (possible, follows)
            return possible, follows
        else:
            possible, follows = self.cache[(conclusion, syllogism)]
            return possible, follows
