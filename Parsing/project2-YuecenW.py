#!/usr/bin/env python
# coding: utf-8

# In[2]:


# class for each cell in the matrix
class Cell:
    def __init__ (self, rule, comp1, comp2, terminal):
        self.rule = rule
        self.comp1 = comp1
        self.comp2 = comp2
        self.terminal = terminal
        
def cky_parser(grammar, words):
    words_count = len(words)
    matrix = []
    #for rows
    for i in range(words_count, 0, - 1):
        matrix.append([[] for x in range(words_count + 1)])
    #for columns
    for j in range(1, words_count + 1):
        each_word = words[j-1]
        if each_word in grammar:
            for rule in grammar[each_word]:
                each_cell = Cell(rule=rule, comp1=(each_word, None), comp2=(None, None), terminal=True)
                matrix[j-1][j].append(each_cell)
        else:
            return None
        for row in range(j-2, -1, -1):
            for k in range(row+1, j):
                for cell_r in matrix[row][k]:
                    for cell_c in matrix[k][j]:
                        const = cell_r.rule + " " + cell_c.rule
                        if const in grammar:
                            for rule in grammar[const]:
                                each_cell = Cell(rule=rule, comp1=(cell_r.rule, cell_r), comp2=(cell_c.rule, cell_c), terminal=False)
                                matrix[row][j].append(each_cell)
            
    return matrix

def display_cell(table, cell):
    if cell.terminal == True:
        print("[" + cell.rule + " " + cell.comp1[0] + "]", end="")
    else:
        print("[" + cell.rule + " ", end="")
        display_cell(table, cell.comp1[1])
        display_cell(table, cell.comp2[1])
        print("]", end="")

def display_cell_parse(table, cell,count):
    if cell.terminal == True:
        for i in range(count):
            print("   ", end="")
        print("[" + cell.rule + " " + cell.comp1[0] + "]")
        
    else:
        for i in range(count):
            print("   ", end="")
        print("[" + cell.rule + " ", end="\n")
        display_cell_parse(table, cell.comp1[1],count+1)
        display_cell_parse(table, cell.comp2[1],count+1)
        for i in range(count):
            print("   ", end="")
        print("]")
        
def display_result(result):
    cells = result[0][-1]
    if len(cells) == 0:
        print("NO VALID PARSES")
    else:
        for i in range(len(cells)):
            print("Valid Parse #" + str(i+1) + ":", end="\n")
            display_cell(result, cells[i])
            print("")
    print("")
    print("Number of valid parses: ", len(cells))
    
def display_parse_tree(result):
    cells = result[0][-1]
    if len(cells) == 0:
        print("NO VALID PARSES")
    else:
        for i in range(len(cells)):
            print("Valid Parse #" + str(i+1) + ":", end="\n")
            display_cell_parse(result, cells[i],0)
            print("")
            
if __name__ == "__main__":
    grammar_path = input("Please enter the CNF grammar file path:")
    grammar = {}
    with open(grammar_path, 'r') as f:
        print("Loading grammar...")
        for line in f:
            line = line.replace("\n", "")
            left, right = line.split(" --> ")
            if right not in grammar:
                grammar.update({right: [left]})
            else:
                grammar[right].append(left)
    while True:
        sentence = input("Please enter a sentence for parser(Type quit for exiting)：")
        if sentence == "quit":
            print("Goodbye!")
            break
        words = sentence.split(" ")
        result = cky_parser(grammar, words)
        if result is None:
            print("NO VALID PARSES")
        elif len(result[0][len(words)]) == 0:
            print("NO VALID PARSES")
        else:
            print("VALID SENTENCE")
            tree = input ("Do you want textual parse trees to be displayed (y/n)?")
            if tree == "n" or tree =="N":
                display_result(result)
            elif tree == "y" or tree == "Y":
                display_result(result)
                display_parse_tree(result)
            else:
                print("Please enter either y/n")
                break

