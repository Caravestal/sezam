from typing import Any, Callable

class BinaryNode:
    value: Any
    left_child: 'BinaryNode'
    right_child: 'BinaryNode'
    parent: 'BinaryNode'

    def __init__(self, data, left=None, right=None):
        self.value = data
        self.left_child = left
        self.right_child = right
        self.parent = None

    def is_leaf(self):
        if self.left_child or self.right_child:
            return False
        else:
            return True

    def add_left_child(self, value: Any):
        self.left_child = BinaryNode(value)
        self.left_child.parent = self

    def add_right_child(self, value: Any):
        self.right_child = BinaryNode(value)
        self.right_child.parent = self

    def traverse_in_order(self, visit: Callable[[Any], None]):
        if self.left_child:
            self.left_child.traverse_in_order(visit)
        visit(self)
        if self.right_child:
            self.right_child.traverse_in_order(visit)

    def traverse_post_order(self, visit: Callable[[Any], None]):
        if self.left_child:
            self.left_child.traverse_post_order(visit)
        if self.right_child:
            self.right_child.traverse_post_order(visit)
        visit(self)

    def traverse_pre_order(self, visit: Callable[[Any], None]):
        visit(self)
        if self.left_child:
            self.left_child.traverse_pre_order(visit)
        if self.right_child:
            self.right_child.traverse_pre_order(visit)

    def __str__(self):
        return str(self.value)

class BinaryTree:
    root: BinaryNode

    def __init__(self, Node):
        self.root = Node

    def traverse_in_order(self, visit: Callable[[Any], None]):
        self.root.traverse_in_order(visit)

    def traverse_post_order(self, visit: Callable[[Any], None]):
        self.root.traverse_post_order(visit)

    def traverse_pre_order(self, visit: Callable[[Any], None]):
        self.root.traverse_pre_order(visit)

t1 = []
def c1(tree: BinaryNode, first: BinaryNode):
    if len(t1)!=0:
        if t1[len(t1)-1] != first:
            t1.append(tree)
    else:
        t1.append(tree)
    if tree.left_child:
        c1(tree.left_child, first)
    if tree.right_child:
        c1(tree.right_child, first)

def c2(second: BinaryNode):
    for i in range(len(t1)):
        if t1[i] == second:
            return second
    return c2(second.parent)

def closet_parent(tree: BinaryTree, first_node: BinaryNode, second_node: BinaryNode):
    c1(tree.root, first_node)
    return c2(second_node)


# drzewo z projektu 2
bn = BinaryNode(1)
bn.add_left_child(2)
bn.left_child.add_left_child(4)
bn.left_child.add_right_child(5)
bn.left_child.left_child.add_left_child(8)
bn.left_child.left_child.add_right_child(9)
bn.add_right_child(3)
bn.right_child.add_right_child(7)
bt = BinaryTree

# sprawdzenie z labów 5
nd = BinaryNode(12)
nd.add_right_child(2)
nd.add_left_child(1)
nd.left_child.add_left_child(1)
nd.right_child.add_right_child(1)

tree = BinaryTree(nd)

# assert tree.root.value == 10
#
# assert tree.root.right_child.value == 2
# assert tree.root.right_child.is_leaf() is False
#
# assert tree.root.left_child.left_child.value == 1
# assert tree.root.left_child.left_child.is_leaf() is True

bt = BinaryTree(bn)
print(closet_parent(bt, bn.left_child.left_child.left_child, bn.right_child.right_child))
print(closet_parent(bt, bn.left_child.left_child.left_child, bn.left_child.right_child))
