class Node():
    def __init__(self, person, movie, parent):
        self.person = person
        self.movie = movie
        self.parent = parent


class QueueFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_person(self, person):
        return any(node.person == person for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node

    
