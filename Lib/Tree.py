'''
Implementation of an n-ary tree
'''
import tempfile
import os
import sqlite3


class Node:
    def __init__(self, top=[], bottom=[], depth=0, parent_id=None, id=None):
        self.id = id
        self.parent_id = parent_id
        # Candidate top and bottom that have been found
        self.top = top
        self.bottom = bottom
        self.depth = depth
        self.children = []

    def get_top(self):
        return self.top

    def get_children(self):
        return self.children

    def get_bottom(self):
        return self.bottom

    def get_depth(self):
        return self.depth

    def __str__(self):
        return 'id = {0}, top= {1}, bottom = {2}, depth = {3}'.format(self.id, self.top, self.bottom, self.depth)


class RegularTree:
    def __init__(self):
        self.node_id_counter = 0
        self.root = Node(
            [],
            [],
            0,
            -1,
            self.get_node_id()
        )

    def get_root(self):
        return self.get_node(0)

    def get_top_of_node(self, id):
        n = self.get_node(id)
        return n.get_top()

    def get_bottom_of_node(self, id):
        n = self.get_node(id)
        return n.get_bottom()

    def get_depth_of_node(self, id):
        n = self.get_node(id)
        return n.get_depth()

    def get_node(self, id):
        node = self.__get_node(self.root, id)
        return node

    def __get_node(self, node, id):
        if node.id == id:
            return node
        else:
            for child in node.children:
                n = self.__get_node(child, id)
                if n:
                    return n

    def __get_parent_id_of(self, child_id):
        child = self.get_node(child_id)
        parent_id = child.parent_id
        return parent_id

    def __get_parent_node_of(self, child_id):
        parent_id = self.__get_parent_id_of(child_id)
        return self.get_node(parent_id)

    def get_leafs(self):
        leafs = []
        leafs = self.__get_leafs(self.root, leafs)
        return leafs

    def __get_leafs(self, node, leafs):
        if len(node.children) == 0:
            leafs.append(node)
        else:
            for child in node.children:
                self.__get_leafs(child, leafs)
        return leafs

    def add_node(self, node):
        self.add_node(node.top, node.bottom, node.depth, node.parent_id, node.id)

    def add_node(self, top, bottom, depth, parent_id, id):
        new_node = Node(top, bottom, depth, parent_id, id)
        parent = self.get_node(parent_id)
        parent.children.append(new_node)

    def get_node_id(self):
        curr = self.node_id_counter
        self.node_id_counter += 1
        return curr

    def get_edge_id(self):
        curr = self.edge_id_counter
        self.edge_id_counter += 1
        return curr

    def dispose(self):
        del self.root


class SQLiteTree:
    def __init__(self):
        self.node_id_counter = 0
        self.edge_id_counter = 0
        self.fname = tempfile.gettempdir() + os.sep + next(tempfile._get_candidate_names()) + '-tree.db'
        self.db = sqlite3.connect(self.fname)
        self.db.execute(
            'create table Nodes (Id INT PRIMARY KEY NOT NULL, Top TEXT NOT NULL, Bottom TEXT NOT NULL, Depth INT NOT NULL);')
        self.db.execute(
            'create table Edges (Id INT PRIMARY KEY NOT NULL, ParentId INT NOT NULL, ChildId INT NOT NULL);')
        self.db.execute(
            "insert into Nodes (Id, Top, Bottom, Depth) values ({0}, {1}, {2}, {3});".format(self.get_node_id(),
                                                                                             '\'\'', '\'\'', 0))
        self.db.execute('PRAGMA journal_mode = OFF;')
        # self.db.execute('PRAGMA synchronous = 0;')
        self.db.execute('PRAGMA cache_size = 1000000;')
        self.db.execute('PRAGMA locking_mode = EXCLUSIVE;')
        self.db.execute('PRAGMA temp_store = MEMORY;')
        self.db.commit()

    def get_root(self):
        return self.get_node(0)

    def get_top_of_node(self, id):
        n = self.get_node(id)
        return [item for item in n.top.split('/') if item]

    def get_bottom_of_node(self, id):
        n = self.get_node(id)
        return [item for item in n.bottom.split('/') if item]

    def get_depth_of_node(self, id):
        c = self.db.execute('select * from Nodes where Id = {0}'.format(id))
        r = next(c)
        _, _, _, depth = (r[0], r[1], r[2], r[3])
        return depth

    def get_node(self, id):
        c = self.db.execute('select * from Nodes where Id = {0}'.format(id))
        r = next(c)
        id, top, bottom, depth = (r[0], r[1], r[2], r[3])

        try:
            getp_cmd = self.db.execute('select * from Edges where ChildId = {0}'.format(id))
            p_r = next(getp_cmd)
            parent_id = p_r[1]
            return Node(top, bottom, depth, parent_id, id)
        except:
            return Node(top, bottom, depth, -1, id)

    def __get_parent_id_of(self, child_id):
        c = self.db.execute('select * from Edges where ChildId = {0}'.format(child_id))
        r = next(c)
        parent_id = r[1]
        return parent_id

    def __get_parent_node_of(self, child_id):
        parent_id = self.__get_parent_id_of(child_id)
        return self.get_node(parent_id)

    def get_leafs(self):
        c = self.db.execute('select * from Nodes where Id not in (select ParentId from Edges);')
        leafs = []
        for r in c:
            id, top, bottom, depth = (r[0], r[1], r[2], r[3])
            self.__get_parent_node_of(id)
            top =  [item for item in self.top.split('/') if item]
            bottom =  [item for item in self.bottom.split('/') if item]
            leafs.append(Node(top, bottom, depth, self.__get_parent_id_of(id), id))
        return leafs

    def add_node(self, node):
        self.add_node(node.top, node.bottom, node.depth, node.parent_id, node.id)

    def add_node(self, top, bottom, depth, parent_id, id):
        top = '/'.join(top)
        bottom = '/'.join(bottom)
        cmd1 = "insert into Nodes (Id, Top, Bottom, Depth) \
              VALUES ({0}, \"{1}\", \"{2}\", {3});".format(id, top, bottom, depth)
        self.db.execute(cmd1)
        cmd2 = "insert into Edges (Id, ParentId, ChildId) values ({0}, {1}, {2});".format(self.get_edge_id(),
                                                                                          parent_id, id)
        self.db.execute(cmd2)
        self.db.commit()

    def get_node_id(self):
        curr = self.node_id_counter
        self.node_id_counter += 1
        return curr

    def get_edge_id(self):
        curr = self.edge_id_counter
        self.edge_id_counter += 1
        return curr

    def dispose(self):
        try:
            self.db.close()
            os.unlink(self.fname)
        except Exception as err_msg:
            print(err_msg)


class RegularStack:
    def __init__(self):
        self.data = []

    def is_empty(self):
        return len(self.data) == 0

    def push(self, record):
        self.data.append(record)

    def pop(self):
        return self.data.pop()

    def dispose(self):
        self.data = []


class SQLiteStack:
    def __init__(self):
        self.fname = tempfile.gettempdir() + os.sep + next(tempfile._get_candidate_names()) + '-stack.db'
        self.db = sqlite3.connect(self.fname)
        self.id = 0
        self.db.execute(
            'create table Records (Id INT PRIMARY KEY NOT NULL, Top TEXT NOT NULL, Bottom TEXT NOT NULL, NodeId INT NOT NULL);')
        self.db.execute('PRAGMA journal_mode = OFF;')
        # self.db.execute('PRAGMA synchronous = 0;')
        self.db.execute('PRAGMA cache_size = 1000000;')
        self.db.execute('PRAGMA locking_mode = EXCLUSIVE;')
        self.db.execute('PRAGMA temp_store = MEMORY;')
        self.db.commit()

    def is_empty(self):
        c = self.db.execute('select Id from Records;')
        return len(c.fetchall()) == 0

    def push(self, record):
        self.db.execute(
            'insert into Records (Id, Top, Bottom, NodeId) values ({0}, \'{1}\', \'{2}\', {3});'.format(self.id,
                                                                                                        '/'.join(
                                                                                                            record[0]),
                                                                                                        '/'.join(
                                                                                                            record[1]),
                                                                                                        record[2]))
        self.id += 1
        self.db.commit()

    def pop(self):
        c = self.db.execute('select * from Records where Id in (select MAX(Id) from Records);')
        r = next(c)
        self.id -= 1
        rresult = [[item for item in r[1].split('/') if item], [item for item in r[2].split('/') if item], r[3]]
        self.db.execute('delete from Records where Id = {0}'.format(r[0]))
        self.db.commit()
        return rresult

    def dispose(self):
        try:
            self.db.close()
            os.unlink(self.fname)
        except Exception as err_msg:
            print(err_msg)


if __name__ == '__main__':
    t = SQLiteTree()
    root = t.get_root()
    t.add_node(top=['a'], bottom=['a'], depth=root.get_depth(), parent_id=root.id, id=t.node_id_counter)
    print(root)
    print([leaf.__str__() for leaf in t.get_leafs()])

    s = SQLiteStack()
    s.push(['a', 'a', 1])
    print(s.pop())
