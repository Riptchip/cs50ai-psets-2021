import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        # Update `self.domains` such that each variable is node-consistent.
        # (Remove any values that are inconsistent with a variable's unary
        #  constraints; in this case, the length of the word.)

        for variable in self.crossword.variables:
            for word in self.domains[variable].copy():
                if len(word) != variable.length and word in self.domains[variable]:
                    self.domains[variable].remove(word)

    def revise(self, x, y):
        # Make variable `x` arc consistent with variable `y`.
        # To do so, remove values from `self.domains[x]` for which there is no
        # possible corresponding value for `y` in `self.domains[y]`.

        # Return True if a revision was made to the domain of `x`; return
        # False if no revision was made.
        
        revised = False
        
        for word in self.domains[x].copy():
            if word in self.domains[x]:
                overlap = self.crossword.overlaps[x, y]

                if overlap == None:
                    if len(self.domains[y]) == 1 and word in self.domains[y]:
                        # Not consistent because the only assignable word for y variable is the same as the word we're checking
                        self.domains[x].remove(word)
                        revised = True
                else:
                    isPossible = False  # True means that the word we're checking do have a possible corresponding value for y variable, False means that the current word cause arc inconsistency
                    
                    for value in self.domains[y]:
                        if word[overlap[0]] == value[overlap[1]] and word != value:
                            isPossible = True

                    if not isPossible:
                        self.domains[x].remove(word)
                        revised = True
            
        return revised

    def ac3(self, arcs=None):
        # Update `self.domains` such that each variable is arc consistent.
        # If `arcs` is None, begin with initial list of all arcs in the problem.
        # Otherwise, use `arcs` as the initial list of arcs to make consistent.

        # Return True if arc consistency is enforced and no domains are empty;
        # return False if one or more domains end up empty.

        queue = list()

        if arcs != None:
            queue = arcs
        else:
            for x in self.crossword.variables:
                for y in self.crossword.variables:
                    if x != y:
                        queue.append((x, y))

        while True:
            x = queue[0][0]
            y = queue[0][1]
            queue.remove(queue[0])

            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False

                # If revise method update `self.domains[x]` add all arcs which have the x variable to the queue to ensure arc consistency
                for var in self.crossword.variables:
                    if var.__eq__(x):
                        continue
                    
                    queue.append((x, var))

            elif len(queue) == 0:
                # It will only be arc consistent when the revise method doesn't update `self.domains[x]` and the queue is empty
                return True

    def assignment_complete(self, assignment):
        # Return True if `assignment` is complete (i.e., assigns a value to each
        # crossword variable); return False otherwise.
        
        for variable in assignment:
            if assignment[variable] == None:
                return False

        return True

    def consistent(self, assignment):
        # Return True if `assignment` is consistent (i.e., words fit in crossword
        # puzzle without conflicting characters); return False otherwise.
        
        for variable in assignment:
            if assignment[variable] == None:
                continue
            
            for var in assignment:
                if var != variable and assignment[var] == assignment[variable]:
                    # Not consistent because both varibles have the same value
                    return False
            
            if variable.length != len(assignment[variable]):
                # Not consistent because the value has the incorrect length
                return False
            
            for neighbor in self.crossword.neighbors(variable):
                if neighbor in assignment:
                    overlap = self.crossword.overlaps[variable, neighbor]
                    if assignment[variable][overlap[0]] != assignment[neighbor][overlap[1]]:
                        # Not consistent because both values don't have the same letter in their overlap place
                        return False

        return True  # If none of the above inconsistencies exists, the assignment is consistent

    def order_domain_values(self, var, assignment):
        # Return a list of values in the domain of `var`, in order by
        # the number of values they rule out for neighboring variables.
        # The first value in the list, for example, should be the one
        # that rules out the fewest values among the neighbors of `var`.

        heuristics = dict()  # Dictionary where keys are values in the domain of `var` and values are the number of values they will rule out for neighboring variables

        for value in self.domains[var]:
            heuristics[value] = 0
            for variable in self.crossword.neighbors(var):
                if variable in assignment:
                    continue

                overlap = self.crossword.overlaps[var, variable]

                for word in self.domains[variable]:
                    if word == value:
                        # Rule out this word because both variables can't have the same value
                        heuristics[value] += 1
                        continue

                    if not value[overlap[0]] == word[overlap[1]]:
                        # Rule out this word because both values don't have the same letter in the overlap place
                        heuristics[value] += 1
        
        return dict(sorted(heuristics.items(), key=lambda item: item[1])).keys()

    def select_unassigned_variable(self, assignment):
        # Return an unassigned variable not already part of `assignment`.
        # Choose the variable with the minimum number of remaining values
        # in its domain. If there is a tie, choose the variable with the highest
        # degree. If there is a tie, any of the tied variables are acceptable
        # return values.

        heuristics = dict()  # Dictionary where keys are unassigned variables and values are the number of neighbors they have
        lowestHeuristic = 10000

        for var in self.crossword.variables:
            if var not in assignment:
                heuristic = len(self.domains[var])

                # If this variable has the lowest heuristic value restart `heuristics` with it
                if heuristic < lowestHeuristic:  
                    heuristics = dict()
                    heuristics[var] = len(self.crossword.neighbors(var))
                    lowestHeuristic = heuristic
                # If this variable has the same heuristic value as other(s) varible(s) with the lowest heuristic value add it in `heuristics`
                elif heuristic == lowestHeuristic:
                    heuristics[var] = len(self.crossword.neighbors(var))
        
        if len(heuristics) == 1:  # If there's no tie between variables in the first heuristic values return the variible with the minimum number of remaining values in its domain
            return list(heuristics.keys())[0]
        elif len(heuristics) > 1:  # If there's a tie, return the varible with the highest nunber if neighbors
            return list(dict(sorted(heuristics.items(), key=lambda item: item[1], reverse=True)).keys())[0]

        return None

    def backtrack(self, assignment):
        # Using Backtracking Search, take as input a partial assignment for the
        # crossword and return a complete assignment if possible to do so.

        # `assignment` is a mapping from variables (keys) to words (values).

        # If no assignment is possible, return None.

        if len(assignment) == len(self.crossword.variables):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            newAssignment = assignment.copy()
            newAssignment[var] = value

            # If value is node consistent with the assignment check arc consistency of all arcs that contains the selected variable assuming that the value are assigned to the variable
            if self.consistent(newAssignment):
                backup = self.domains[var].copy()
                self.domains[var] = [value]

                arcs = list()

                for y in self.crossword.variables:
                    if var != y:
                        arcs.append((var, y))

                # If have arc consistency call recurssively backtrack with the new assignment, else try another value
                if self.ac3(arcs):
                    result = self.backtrack(newAssignment)

                    # If backtrack return a solution return it, else try another value
                    if result != None:
                        return result
                    else:
                        self.domains[var] = backup.copy()
                else:
                    self.domains[var] = backup.copy()
        
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
