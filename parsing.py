# Various functions to enable parsing of sentences
# from our artificial grammars

# Load part-of-speech labels
posDict = {}
fi = open("pos.txt", "r") # MIGHT NEED TO CHANGE BACK
for line in fi:
    parts = line.split("\t")
    posDict[parts[0].strip()] = parts[1].strip()

posDict2 = {}
fi = open("pos2.txt", "r")
for line in fi:
    parts = line.split("\t")
    posDict2[parts[0].strip()] = parts[1].strip()


# Conert a sentence to part-of-speech tags
def sent_to_pos(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        if word in posDict:
            pos_tags.append(posDict[word])
        else:
            pos_tags.append(posDictTense[word])

    return pos_tags

# Convert a sentence to part-of-speech tags
# from the second part-of-speech file
def sent_to_posb(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        pos_tags.append(posDict2[word])

    return pos_tags

# Convert a sequence of part-of-speech tags into
# a parse. Works by successively grouping together
# neighboring pairs.
def pos_to_parse(pos_seq):
    full_parse = []

    current_parse = []
    current_nodes = pos_seq

    new_nodes = []
    skip_next = 0

    while len(current_nodes) > 1:
        for index, node in enumerate(current_nodes):
            if skip_next:
                skip_next = 0
                continue
            if node == "D" and current_nodes[index + 1] == "N":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "P" and current_nodes[index + 1] == "NP":
                new_nodes.append("PP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "P" and current_nodes[index + 1] == "NP_f":
                new_nodes.append("PP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "PP":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "RC":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "NP_f":
                new_nodes.append("VP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "T":
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "A": 
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "VP": 
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "T": 
                new_nodes.append("NP_f")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("S")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "S" and current_nodes[index + 1] == "T":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "A" and current_nodes[index + 1] == "S_bar": 
                new_nodes.append("A_S_bar") 
                current_parse.append([index, index + 1]) 
                skip_next = 1 
            elif node == "K" and current_nodes[index + 1] == "ROOT":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "K" and current_nodes[index + 1] == "ROOT":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "A" and current_nodes[index + 1] == "VP":
                new_nodes.append("VP_f")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "C" and current_nodes[index + 1] == "VP":
                new_nodes.append("VP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "R" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("RC")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "R" and current_nodes[index + 1] == "S":
                new_nodes.append("RC")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "A":
                new_nodes.append("NP_f")
                current_parse.append([index])
            elif node == "NP_f" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("S")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "NP_bar":
                new_nodes.append("VP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP_bar" and current_nodes[index + 1] == "VP":
                new_nodes.append("S_bar")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "VP":
                new_nodes.append("NP_bar")
                current_parse.append([index])
            elif node == "A_S_bar" and current_nodes[index + 1] == "T":
                new_nodes.append("ROOT") # CHANGE
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "ROOT" and current_nodes[index + 1] == "G":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            else:
                new_nodes.append(node)
                current_parse.append([index])


        current_nodes = new_nodes
        new_nodes = []
        skip_next = 0
        full_parse.append(current_parse)
        current_parse = []

    full_parse.append([[0]])

    return full_parse

# Parse a sentence from the question formation dataset
def parse_question(sent):
    return pos_to_parse(sent_to_posb(sent))

# Create a part-of-speech dictionary for tense reinflection sentences
posDictTense = {}
fi = open("pos_tense.txt", "r")
for line in fi:
    parts = line.split("\t")
    posDictTense[parts[0].strip()] = parts[1].strip()

# Convert a tense reinflection sentence into
# a sequence of part-of-speech tags
def sent_to_pos_tense(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        pos_tags.append(posDictTense[word])

    return pos_tags

# Convert a sequence of part-of-speech tags into a parse.
# Works by successively grouping together
# neighboring pairs.
def pos_to_parse_tense(pos_seq):
    full_parse = []

    current_parse = []
    current_nodes = pos_seq

    new_nodes = []
    skip_next = 0

    while len(current_nodes) > 1:
        for index, node in enumerate(current_nodes):
            if skip_next:
                skip_next = 0
                continue
            if node == "D" and current_nodes[index + 1] == "N":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "P" and current_nodes[index + 1] == "NP":
                new_nodes.append("PP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "P" and current_nodes[index + 1] == "NP_f":
                new_nodes.append("PP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "PP":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "RC":
                new_nodes.append("NP")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "NP_f":
                new_nodes.append("VP_f")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "NP" and current_nodes[index + 2] == "VP_f":
                new_nodes.append("VP_f")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "T":
                new_nodes.append("VP_f")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("VP_f")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "T":
                new_nodes.append("NP_f")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("S")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "S" and current_nodes[index + 1] == "T":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "K" and current_nodes[index + 1] == "ROOT":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "K" and current_nodes[index + 1] == "ROOT":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "R" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("RC")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "R" and current_nodes[index + 1] == "S":
                new_nodes.append("RC")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "A":
                new_nodes.append("NP_f")
                current_parse.append([index])
            elif node == "NP_f" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("S")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "V" and current_nodes[index + 1] == "NP_bar":
                new_nodes.append("VP_f")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP_bar" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("S_bar")
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "NP" and current_nodes[index + 1] == "VP_f":
                new_nodes.append("NP_bar")
                current_parse.append([index])
            elif node == "A_S_bar" and current_nodes[index + 1] == "T":
                new_nodes.append("ROOT") 
                current_parse.append([index, index + 1])
                skip_next = 1
            elif node == "ROOT" and current_nodes[index + 1] == "G":
                new_nodes.append("ROOT")
                current_parse.append([index, index + 1])
                skip_next = 1
            else:
                new_nodes.append(node)
                current_parse.append([index])

        current_nodes = new_nodes
        new_nodes = []
        skip_next = 0
        full_parse.append(current_parse)
        current_parse = []

    full_parse.append([[0]])

    return full_parse

# Parse a sentence from the tense reinflection dataset
def parse_tense(sent):
    return pos_to_parse_tense(sent_to_pos_tense(sent))




