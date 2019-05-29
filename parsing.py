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


def preprocess(sent):
    words = sent.split()

    if words[-1] == "quest" or words[-1] == "QUEST":
        if words[-2] == ".":
            return " ".join(["+Q"] + words[:-1])
        else:
            return " ".join(insert_trace(words)[:-1])

    elif words[-1] == "decl" or words[-1] == "DECL":
        return " ".join(["-Q"] + words[:-1])

    else:
        return sent

def insert_trace(words):
    pos_tags = []
    for word in words:
        pos_tags.append(posDict[word])

    for index, tag in enumerate(pos_tags):
        if tag == "V" and pos_tags[index - 1] != "A":
            trace_index = index

    return words[:trace_index] + ["t"] + words[trace_index:]

def sent_to_pos_tree(sent):
    sent = preprocess(sent)
    words = sent.split()

    pos_tags = []

    for word in words:
        pos_tags.append(posDict[word])

    return pos_tags

def sent_to_pos(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        if word in posDict:
            pos_tags.append(posDict[word])
        else:
            pos_tags.append(posDictTense[word])

    return pos_tags



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
            elif node == "V" and current_nodes[index + 1] == "T": # turn V to VP
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "A": # turn V to VP
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "VP": # turn V to VP
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "T": # turn V to VP
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
            elif node == "S_bar" and current_nodes[index + 1] == "T": # CHANGE
                new_nodes.append("S_bar_punc") # CHANGE
                current_parse.append([index, index + 1]) # CHANGE
                skip_next = 1 # CHANGE
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
            elif node == "A" and current_nodes[index + 1] == "S_bar_punc":
                new_nodes.append("ROOT") # CHANGE
                current_parse.append([index, index + 1])
                skip_next = 1

            else:
                new_nodes.append(node)
                current_parse.append([index])


        #print(new_nodes)
        current_nodes = new_nodes
        new_nodes = []
        skip_next = 0
        full_parse.append(current_parse)
        current_parse = []

    full_parse.append([[0]])

    return full_parse

def sent_to_posb(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        pos_tags.append(posDict2[word])

    return pos_tags


def pos_to_parseb(pos_seq):
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
            elif node == "V" and current_nodes[index + 1] == "T": # turn V to VP
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "A": # turn V to VP
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "VP": # turn V to VP
                new_nodes.append("VP")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "T": # turn V to VP
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
            elif node == "A" and current_nodes[index + 1] == "S_bar": # CHANGE
                new_nodes.append("A_S_bar") # CHANGE
                current_parse.append([index, index + 1]) # CHANGE
                skip_next = 1 # CHANGE
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


        #print(new_nodes)
        current_nodes = new_nodes
        new_nodes = []
        skip_next = 0
        full_parse.append(current_parse)
        current_parse = []

    full_parse.append([[0]])

    return full_parse


def parse_nopre(sent):
	return pos_to_parseb(sent_to_posb(sent))

posDictTense = {}
fi = open("pos_tense.txt", "r")
for line in fi:
        parts = line.split("\t")
        posDictTense[parts[0].strip()] = parts[1].strip()

def sent_to_pos_tense(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        pos_tags.append(posDictTense[word])

    return pos_tags

def pos_to_parse_tense(pos_seq):
    full_parse = []

    current_parse = []
    current_nodes = pos_seq

    new_nodes = []
    skip_next = 0

    while len(current_nodes) > 1:
        for index, node in enumerate(current_nodes):
            #print(node)
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
            elif node == "V" and current_nodes[index + 1] == "T": # turn V to VP
                new_nodes.append("VP_f")
                current_parse.append([index])
            elif node == "V" and current_nodes[index + 1] == "VP_f": # turn V to VP
                new_nodes.append("VP_f")
                current_parse.append([index])
            elif node == "NP" and current_nodes[index + 1] == "T": # turn V to VP
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


#        print(new_nodes)
        current_nodes = new_nodes
        new_nodes = []
        skip_next = 0
        full_parse.append(current_parse)
        current_parse = []

    full_parse.append([[0]])

    return full_parse

def parse_tense(sent):
    return pos_to_parse_tense(sent_to_pos_tense(sent))




