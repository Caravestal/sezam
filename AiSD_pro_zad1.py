from typing import Any

class Node:
    value: Any
    next: 'Node'

class LinkedList:
    head: Node
    tail: Node

    def __init__(self):
        self.head = None
        self.tail = None

    def push(self, value: Any) -> None:
        temp = Node()
        temp.value = value
        temp.next = self.head
        self.head = temp

    def append(self, value: Any) -> None:
        temp = Node
        temp.value = value
        temp.next = self.tail
        self.head = temp

    def node(self, at: int) -> Node:
        n = self.head
        for x in range(at):
            n = n.next
        return n

    def insert(self, value: Any, after: Node) -> None:
        value.next = after.next
        after.next = value

    def pop(self) -> Any:
        temp = self.head
        self.head = self.head.next
        return temp

    def remove_last(self) -> Any:
        temp = self.head
        while(temp.next.next != None):
            temp = temp.next
        tem = temp.next
        temp.next = None
        self.tail = temp
        return tem

    def remove(self, after: Node) -> Any:
        after.next = after.next.next

    def print(self):
        print(str(self.head.value) + " -> " + str(self.head.next.value))

    def __len__(self):
        wyn = 0
        temp = self.head
        while(temp != None):
            temp = temp.next
            wyn = wyn + 1
        return wyn
