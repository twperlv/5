class TreeNode:
    def __init__(self, key, value, left=None, right=None, parent=None):
        self.key = key
        self.payload = value
        self.left_child = left
        self.right_child = right
        self.parent = parent

    def has_left_child(self):
        return self.left_child is not None

    def has_right_child(self):
        return self.right_child is not None

    def is_left_child(self):
        return self.parent and self.parent.left_child == self

    def is_right_child(self):
        return self.parent and self.parent.right_child == self

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return not (self.right_child or self.left_child)

    def has_any_child(self):
        return self.right_child or self.left_child

    def has_both_children(self):
        return self.right_child and self.left_child

    def replace_node_data(self, key, value, left_child, right_child):
        self.key = key
        self.payload = value
        self.left_child = left_child
        self.right_child = right_child
        if self.has_left_child():
            self.left_child.parent = self
        if self.has_right_child():
            self.right_child.parent = self


class BinarySearchTree:
    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def __iter__(self):
        if self.root:
            return iter(self.root)
        return iter([])

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None:
            return None
        if key == node.key:
            return node
        elif key < node.key:
            return self._search(node.left_child, key)
        else:
            return self._search(node.right_child, key)

    def insert(self, key, value):
        if self.root is None:
            self.root = TreeNode(key, value)
        else:
            self._insert(self.root, key, value)
        self.size += 1

    def _insert(self, node, key, value):
        if key == node.key:
            node.payload = value
        elif key < node.key:
            if node.has_left_child():
                self._insert(node.left_child, key, value)
            else:
                node.left_child = TreeNode(key, value, parent=node)
        else:
            if node.has_right_child():
                self._insert(node.right_child, key, value)
            else:
                node.right_child = TreeNode(key, value, parent=node)

    def delete(self, key):
        node = self.search(key)
        if node:
            self._delete(node)
            self.size -= 1

    def _delete(self, node):
        if node.is_leaf():  # Case 1: No children
            if node.is_left_child():
                node.parent.left_child = None
            elif node.is_right_child():
                node.parent.right_child = None
            else:  # root node
                self.root = None
        elif node.has_any_child():  # Case 2: One child
            if node.has_left_child():
                child = node.left_child
            else:
                child = node.right_child
            if node.is_left_child():
                node.parent.left_child = child
            elif node.is_right_child():
                node.parent.right_child = child
            else:
                self.root = child
            child.parent = node.parent
        else:  # Case 3: Two children
            successor = self._find_min(node.right_child)
            node.key = successor.key
            node.payload = successor.payload
            self._delete(successor)

    def _find_min(self, node):
        while node.has_left_child():
            node = node.left_child
        return node

    def preorder(self, node):
        if node:
            print(f" {node.key}", end="")
            self.preorder(node.left_child)
            self.preorder(node.right_child)

    def postorder(self, node):
        if node != None:
            self.postorder(node.left_child)
            self.postorder(node.right_child)
            print(f" {node.key}", end="")

    def inorder(self, node):
        if node != None:
            self.inorder(node.left_child)
            print(f" {node.key}", end="")
            self.inorder(node.right_child)

    def print_traversals(self):
        print("Pre-order traversal:", end="")
        self.preorder(self.root)
        print("\nPost-order traversal:", end="")
        self.postorder(self.root)
        print("\nIn-order traversal:", end="")
        self.inorder(self.root)
        print("")

    # Adapted function from J. V. 
    # (https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python)
    def display(self):
        lines, *_ = self._display_aux(self.root)
        for line in lines:
            print(line)

    def _display_aux(self, node):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if node.right_child is None and node.left_child is None:
            line = '%s' % node.key
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if node.right_child is None:
            lines, n, p, x = self._display_aux(node.left_child)
            s = '%s' % node.key
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if node.left_child is None:
            lines, n, p, x = self._display_aux(node.right_child)
            s = '%s' % node.key
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self._display_aux(node.left_child)
        right, m, q, y = self._display_aux(node.right_child)
        s = '%s' % node.key
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2




bst = BinarySearchTree()
a = [49, 38, 65, 97, 60, 76, 13, 27, 5, 1]
b = [149, 38, 65, 197, 60, 176, 13, 217, 5, 11]
c = [49, 38, 65, 97, 64, 76, 13, 77, 5, 1, 55, 50, 24]
d = a + b + c

for value in d:
    bst.insert(value, str(value))

bst.display()
bst.print_traversals()

bst.delete(13)

bst.display()
bst.print_traversals()
