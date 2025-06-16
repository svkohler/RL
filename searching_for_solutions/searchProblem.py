from display import Displayable
import matplotlib.pyplot as plt
import random

class Search_problem(Displayable):
    """A search problem consists of:
    start node
    neighbors function that gives the neighbors of a node
    specification of a goal
    (optional) heuristic function.
    methods must be overridden to define a search problem."""
    def start_node(self):
        """returns start node"""
        raise NotImplementedError("start_node") # abstract method
    def is_goal(self,node):
        """is True if node is a goal"""
        raise NotImplementedError("is_goal") # abstract method
    def neighbors(self,node):
        """returns a list (or enumeration) of the arcs for the neighbors of
        node"""
        raise NotImplementedError("neighbors") # abstract method
    def heuristic(self,n):
        """Gives the heuristic value of node n. Returns 0 if not overridden."""
        return 0
    
class Arc(object):
    """An arc consists of
    a from_node and a to_node node
    a (non-negative) cost
    an (optional) action
    """
    def __init__(self, from_node, to_node, cost=1, action=None):
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost
        assert cost >= 0, (f"Cost cannot be negative: {self}, cost={cost}") 
        self.action = action
    
    def __repr__(self):
        """string representation of an arc""" 
        if self.action:
            return f"{self.from_node} --{self.action}--> {self.to_node}" 
        else:
            return f"{self.from_node} --> {self.to_node}"
        

class Search_problem_from_explicit_graph(Search_problem):
    """A search problem from an explicit graph.
    """ 
    def __init__(self, title, nodes, arcs, start=None, goals=set(), hmap={},positions=None):
        """ A search problem consists of:
        * list or set of nodes
        * list or set of arcs
        * start node
        * list or set of goal nodes
        * hmap: dictionary that maps each node into its heuristic value.
        * positions: dictionary that maps each node into its (x,y) position
        """
        self.title = title
        self.neighs = {}
        self.nodes = nodes
        for node in nodes:
            self.neighs[node]=[]
        self.arcs = arcs
        for arc in arcs:
            self.neighs[arc.from_node].append(arc)
        self.start = start
        self.goals = goals
        self.hmap = hmap
        if positions is None:
            self.positions = {node:(random.random(),random.random()) for node in nodes}
        else:
            self.positions = positions
    def start_node(self): 
        """returns start node""" 
        return self.start
    def is_goal(self,node):
        """is True if node is a goal""" 
        return node in self.goals
    def neighbors(self,node):
        """returns the neighbors of node (a list of arcs)""" 
        return self.neighs[node]
    def heuristic(self,node):
        """Gives the heuristic value of node n. Returns 0 if not overridden in the hmap.""" 
        if node in self.hmap:
            return self.hmap[node] 
        else:
            return 0
    def __repr__(self):
        """returns a string representation of the search problem""" 
        res=""
        for arc in self.arcs:
            res += f"{arc}. " 
        return res
    
    def show(self, fontsize=10, node_color='orange', show_costs = True):
        """Show the graph as a figure
        """
        self.fontsize = fontsize
        self.show_costs = show_costs
        # plt.ion() # interactive
        ax = plt.figure().gca()
        ax.set_axis_off()
        plt.title(self.title, fontsize=fontsize)
        self.show_graph(ax, node_color)
        plt.show()

    def show_graph(self, ax, node_color='orange'):
        bbox = dict(boxstyle="round4,pad=1.0,rounding_size=0.5",facecolor=node_color)
        for arc in self.arcs: 
            self.show_arc(ax, arc)
        for node in self.nodes:
            self.show_node(ax, node, node_color = node_color)

    def show_node(self, ax, node, node_color): 
        x,y = self.positions[node]
        ax.text(x,y,node,bbox=dict(boxstyle="round4,pad=1.0,rounding_size=0.5", facecolor=node_color),
                      ha='center',va='center', fontsize=self.fontsize)
        
    def show_arc(self, ax, arc, arc_color='black', node_color='white'): 
        from_pos = self.positions[arc.from_node]
        to_pos = self.positions[arc.to_node] 
        ax.annotate(arc.to_node, from_pos, xytext=to_pos,
                            arrowprops={'arrowstyle':'<|-', 'linewidth': 2,
                                                'color':arc_color},
        bbox=dict(boxstyle="round4,pad=1.0,rounding_size=0.5", facecolor=node_color),
                                    ha='center',va='center',
                                    fontsize=self.fontsize)
                # Add costs to middle of arcs:
        if self.show_costs:
            ax.text((from_pos[0]+to_pos[0])/2, (from_pos[1]+to_pos[1])/2,
            arc.cost, bbox=dict(pad=1,fc='w',ec='w'), ha='center',va='center',fontsize=self.fontsize)


class Path(object):
    """A path is either a node or a path followed by an arc"""

    def __init__(self,initial,arc=None):
        """initial is either a node (in which case arc is None) or a path (in which case arc is an object of type Arc)""" 
        self.initial = initial
        self.arc=arc
        if arc is None:
            self.cost=0 
        else:
            self.cost = initial.cost+arc.cost
    def end(self):
        """returns the node at the end of the path""" 
        if self.arc is None:
            return self.initial 
        else:
            return self.arc.to_node
    def nodes(self):
        """enumerates the nodes of the path from the last element backwards """
        current = self
        while current.arc is not None:
            yield current.arc.to_node
            current = current.initial 
        yield current.initial
    def initial_nodes(self):
        """enumerates the nodes for the path before the end node. This calls nodes() for the initial part of the path.
        """
        if self.arc is not None:
            yield from self.initial.nodes()

    def __repr__(self):
        """returns a string representation of a path""" 
        if self.arc is None:
            return str(self.initial) 
        elif self.arc.action:
            return f"{self.initial}\n --{self.arc.action}--> {self.arc.to_node}"
        else:
            return f"{self.initial} --> {self.arc.to_node}"


problem1 = Search_problem_from_explicit_graph(
    'Problem 1',
    {'A','B','C','D','G'},
    [Arc('A','B',3), Arc('A','C',1), Arc('B','D',1), Arc('B','G',3),
    Arc('C','B',1), Arc('C','D',3), Arc('D','G',1)],
    start = 'A',
    goals = {'G'},
    positions={'A': (0, 1), 'B': (0.5, 0.5), 'C': (0,0.5),
    'D': (0.5,0), 'G': (1,0)})

problem1.show()




