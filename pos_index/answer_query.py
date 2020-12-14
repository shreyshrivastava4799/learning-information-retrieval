import os
import sys
import pickle
from typing import Tuple


class Node(object):

    def __init__(self, char: str):
        self.char = char
        self.child = []
        self.word_end = False


def add(root, word: str):
    '''
        This function adds new words to the tree.
    '''
    node = root
    for char in word:

        found_in_child = False
        for child in node.child:
            if child.char == char:

                node = child
                found_in_child = True
                break

        if not found_in_child:
            new_node = Node(char)
            node.child.append(new_node)
            node = new_node

    node.word_end = True


def traverse_tree(root, prefix):
    '''
        This function traverses all the words in vocabulary with given prefix.
    '''
    words = []
    if not root.child:
        words.append(prefix)
        return words

    for child in root.child:
        words.extend(traverse_tree(child, prefix + child.char))

    return words


def find_prefix(root, prefix: str):
    '''
        This function helps to find the tree node which represents the given prefix.
    '''
    node = root
    if not root.child:
        return []

    for char in prefix:

        char_not_found = True
        for child in node.child:
            if child.char == char:

                char_not_found = False

                node = child
                break

        if char_not_found:
            return []

    return traverse_tree(node, prefix)

if __name__ == "__main__":

    # read the stored Inverted Positional Index
    with open(os.path.join(os.getcwd(), 'InvPosIdx.pkl'), 'rb') as f:
        InvPosIdx = pickle.load(f)

    outfile = open(os.path.join(os.getcwd(), 'RESULTS1_17CS30034.txt'), 'w')

    # create  Tree for forward and reverse word vocabulary
    forward_root = Node('*')
    reverse_root = Node('*')

    # vocabulary
    vocab = list(InvPosIdx.keys())
    # print(len(vocab ))
    # print(vocab)
    rev_vocab = [word[::-1] for word in vocab]

    # create B-tree
    for word in vocab:
        add(forward_root, word)

    for word in rev_vocab:
        add(reverse_root, word)

    # read the file provided as an argument
    with open(sys.argv[1]) as fp:

        for line in fp:
            q_word = line.strip()
            print(q_word)

            # case 1: mon*
            if q_word[-1] == '*':
                query_words = find_prefix(forward_root, q_word[:-1])

            # case 2: *mon
            elif q_word[0] == '*':
                query_words = find_prefix(reverse_root, q_word[1:][::-1])
                query_words = [word[::-1] for word in query_words]

            # case 3: mo*n
            else:
                prefix, suffix = q_word.split('*')
                for_query_words = find_prefix(forward_root, prefix)
                rev_query_words = find_prefix(reverse_root, suffix[::-1])
                rev_query_words = [word[::-1] for word in rev_query_words]
                query_words = list(set(for_query_words) & set(rev_query_words))

            text = ""
            for word in query_words:
                text += f'{word} :'
                for occurrence in InvPosIdx[word]:
                    filename, positions = occurrence
                    for position in positions:
                        if position == positions[-1]:
                            text += f' <{filename}, {position}>;'
                            break
                        text += f' <{filename}, {position}>,'

            text += '\n'
            outfile.write(text)

    outfile.close()
