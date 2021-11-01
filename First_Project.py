from typing import Any

class Node:
    value: Any
    next: 'Node'

class LinkedList:
    head: Node
    tail: Node

# ad.1

    def __init__(self):
        self.head = None
        self.tail = None

    def push(self, value: Any) -> None:
        temp = Node()
        temp.value = value
        temp.next = self.head
        self.head = temp

    def append(self, value: Any) -> None:
        temp = self.head
        if (temp == None):
            t = Node()
            t.value = value
            t.next = None
            self.head = t
            return
        while (temp.next != None):
            temp = temp.next
        t = Node()
        t.value = value
        t.next = None
        temp.next = t

    def node(self, at: int) -> Node:
        temp = self.head
        for x in range(at):
            temp = temp.next
        return temp

    def insert(self, value: Any, after: Node) -> None:
        temp = Node
        temp.value = value
        temp.next = after.next
        after.next = temp

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

    def __str__(self):
        temp = self.head
        wyn = ""
        while(temp != None):
            wyn += str(temp.value)
            temp = temp.next
            if(temp != None):
                wyn += " -> "
        return wyn

    def __len__(self):
        wyn = 0
        temp = self.head
        while(temp != None):
            temp = temp.next
            wyn = wyn + 1
        return wyn

list_ = LinkedList()

assert list_.head == None

list_.push(1)
list_.push(0)

assert str(list_) == '0 -> 1'

list_.append(9)
list_.append(10)

assert str(list_) == '0 -> 1 -> 9 -> 10'

middle_node = list_.node(at=1)
list_.insert(5, after=middle_node)

assert str(list_) == '0 -> 1 -> 5 -> 9 -> 10'

first_element = list_.node(at=0)
returned_first_element = list_.pop()

assert first_element == returned_first_element

last_element = list_.node(at=3)
returned_last_element = list_.remove_last()

assert last_element == returned_last_element
assert str(list_) == '1 -> 5 -> 9'

second_node = list_.node(at=1)
list_.remove(second_node)

assert str(list_) == '1 -> 5'

# ad.2

class Stack:
    _storage: LinkedList

    def __init__(self):
        self._storage = LinkedList()

    def __len__(self):
        return len(self._storage)

    def push(self, element: Any) -> None:
        return self._storage.push(element)

    def __str__(self):
        temp = self._storage.head
        wyn = ""
        while (temp != None):
            wyn += str(temp.value)
            temp = temp.next
            if (temp != None):
                wyn += "\n"
        return wyn

    def pop(self):
        return self._storage.pop()

stack = Stack()

assert len(stack) == 0

stack.push(3)
stack.push(10)
stack.push(1)

assert len(stack) == 3

top_value = stack.pop()

assert top_value.value == 1

assert len(stack) == 2

print(str(stack))

# ad.3

class Queue:
    _storage: LinkedList

    def __init__(self):
        self._storage = LinkedList()

    def __len__(self):
        return len(self._storage)

    def __str__(self):
        temp = self._storage.head
        t = ""
        while (True):
            t = t + str(temp.value)
            if (temp.next == None):
                break
            temp = temp.next
            t = t + ", "
        return t

    def peak(self) -> Any:
        return self._storage.head.value

    def enqueue(self, element: Any) -> None:
        self._storage.append(element)

    def dequeue(self) -> Any:
        return self._storage.pop()

queue = Queue()

assert len(queue) == 0

queue.enqueue('klient1')
queue.enqueue('klient2')
queue.enqueue('klient3')

assert str(queue) == 'klient1, klient2, klient3'

client_first = queue.dequeue()

assert client_first.value == 'klient1'
assert str(queue) == 'klient2, klient3'
assert len(queue) == 2
