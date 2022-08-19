import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    # Compute and return a joint probability.

    # The probability returned should be the probability that
    #     * everyone in set `one_gene` has one copy of the gene, and
    #     * everyone in set `two_genes` has two copies of the gene, and
    #     * everyone not in `one_gene` or `two_gene` does not have the gene, and
    #     * everyone in set `have_trait` has the trait, and
    #     * everyone not in set` have_trait` does not have the trait.
    
    probability = 0.0
    
    # Dictionary to store variables from the analyzed scenario
    scenario = dict()

    for person in people:
        scenario[person] = dict()
        
        if person in one_gene:
            scenario[person]["gene"] = 1
        elif person in two_genes:
            scenario[person]["gene"] = 2
        else:
            scenario[person]["gene"] = 0

        if person in have_trait:
            scenario[person]["trait"] = True
        else:
            scenario[person]["trait"] = False

        scenario[person]["probability"] = 0.0
    
    # Iterate through all people to attach the probabilities of the given scenario variables
    for person in people:
        # Attaching the probabilities of genes
        if person in one_gene:
            if people[person]["mother"] == None:
                scenario[person]["probability"] = PROBS["gene"][1]
            else:
                get_gene_probability = create_get_gene_probability(scenario, person, people)

                scenario[person]["probability"] = get_gene_probability["mother"][True] * get_gene_probability["father"][False] + get_gene_probability["father"][True] * get_gene_probability["mother"][False]
        elif person in two_genes:
            if people[person]["mother"] == None:
                scenario[person]["probability"] = PROBS["gene"][2]
            else:
                get_gene_probability = create_get_gene_probability(scenario, person, people)

                scenario[person]["probability"] = get_gene_probability["mother"][True] * get_gene_probability["father"][True]
        else:
            if people[person]["mother"] == None:
                scenario[person]["probability"] = PROBS["gene"][0]
            else:
                get_gene_probability = create_get_gene_probability(scenario, person, people)

                scenario[person]["probability"] = get_gene_probability["mother"][False] * get_gene_probability["father"][False]

        # Attaching the probabilities of trait
        if person in have_trait:
            scenario[person]["probability"] *= PROBS["trait"][scenario[person]["gene"]][True]
        else:
            scenario[person]["probability"] *= PROBS["trait"][scenario[person]["gene"]][False]

    # Multiply all attached probabilities to found the probability of this scenario happen
    for person in scenario:
        probability = probability * scenario[person]["probability"] if probability > 0 else scenario[person]["probability"]
        
    return probability


def create_get_gene_probability(scenario, person, people):
    # Create a dictionary to store the probabilities of the mother and father
    # of a given person pass or not a gene that cause the trait on a given scenario
    get_gene_probability = {
        "mother": {
            True: 0.0,
            False: 0.0
        },

        "father": {
            True: 0.0,
            False: 0.0
        }
    }

    # Fill this dictionary with the given informations
    if scenario[people[person]["mother"]]["gene"] == 2:
        get_gene_probability["mother"][True] = 1 - PROBS["mutation"]
        get_gene_probability["mother"][False] = PROBS["mutation"]
    elif scenario[people[person]["mother"]]["gene"] == 1:
        get_gene_probability["mother"][True] = 0.5
        get_gene_probability["mother"][False] = 0.5
    else:
        get_gene_probability["mother"][True] = PROBS["mutation"]
        get_gene_probability["mother"][False] = 1 - PROBS["mutation"]

    if scenario[people[person]["father"]]["gene"] == 2:
        get_gene_probability["father"][True] = 1 - PROBS["mutation"]
        get_gene_probability["father"][False] = PROBS["mutation"]
    elif scenario[people[person]["father"]]["gene"] == 1:
        get_gene_probability["father"][True] = 0.5
        get_gene_probability["father"][False] = 0.5
    else:
        get_gene_probability["father"][True] = PROBS["mutation"]
        get_gene_probability["father"][False] = 1 - PROBS["mutation"]

    return get_gene_probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    # Add to `probabilities` a new joint probability `p`.
    # Each person should have their "gene" and "trait" distributions updated.
    # Which value for each distribution is updated depends on whether
    # the person is in `have_gene` and `have_trait`, respectively.

    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    # Update `probabilities` such that each probability distribution
    # is normalized (i.e., sums to 1, with relative proportions the same).

    for person in probabilities:
        probabilitiesSum = 0.0

        for gene in probabilities[person]["gene"]:
            probabilitiesSum += probabilities[person]["gene"][gene]

        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] /= probabilitiesSum

        probabilitiesSum = 0.0

        probabilitiesSum = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]

        probabilities[person]["trait"][True] /= probabilitiesSum
        probabilities[person]["trait"][False] /= probabilitiesSum


if __name__ == "__main__":
    main()
