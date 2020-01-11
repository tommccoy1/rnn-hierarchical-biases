# Functions for getting finer-grained evaluations of output sentences

# Given a sentence from the question formation generalization set,
# determine the type of relative clause it has modifying the subject:
# an object relative clause (ORC), a transitive subject relative
# clause (SRC_t), or an intransitive subject relative clause (SRC_i).
dets = ["the", "some", "my", "your", "our", "her"]
def rc_category(sent):
    words_pre = sent.split()
    words = []
    for word in words_pre:
        if word != "-q" and word != "-Q" and word != "+q" and word != "+Q" and word != "t":
            words.append(word)

    if words[3] in dets:
        return "ORC"

    if words[5] in dets:
        return "SRC_t"

    else:
        return "SRC_i"



# Categorize the outputs of a model trained on question formation in a way
# inspired by Crain & Nakayama 1987. The outputs are categorized based on whether
# they could have been formed by placing an auxiliary at the front of the 
# input and deleting (or failing to delete) an auxiliary from within the input; if so,
# this function then categorizes them based on which auxiliary was placed at the front and
# which was deleted.
# Most outputs are in the "other" category which cannot be categorized in this way.
# Example: 
# - Input: my walrus that does laugh doesn't giggle.
# - Output: does my walrus that does laugh giggle?
# - Categorization: p1d2 (for "prepose first, delete second")
auxes = ["can", "could", "will", "would", "do", "does", "don't", "doesn't"]
def crain(sentence, output):
    index1 = -1
    index2 = -1

    words_pre = sentence.replace("?", ".").replace("decl", "quest").replace("DECL", "quest").replace("QUEST", "quest").split()

    words = []
    for word in words_pre:
        if word != "-q" and word != "-Q" and word != "+q" and word != "+Q" and word != "t":
            words.append(word)


    for ind, word in enumerate(words):
        if word in auxes:
            if index1 == -1:
                index1 = ind
            elif index2 == -1:
                index2 = ind

    aux1 = words[index1]
    aux2 = words[index2]


    d1 = " ".join(words[:index1] + words[index1 + 1:-1])
    d2 = " ".join(words[:index2] + words[index2 + 1:-1])
    dn = " ".join(words[:-1]) 
    if d1[-1] != ".":
        d1 = d1 + " ."
    if d2[-1] != ".":
        d2 = d2 + " ."
    if dn[-1] != ".":
        dn = dn + " ."

    output = output.replace("?", ".").replace("decl", "quest").replace("DECL", "quest").replace("QUEST", "quest")
    output_words = output.split()
    output_words_new = []

    for word in output_words:
        if word != "-q" and word != "-Q" and word != "+q" and word != "+Q" and word != "t":
            output_words_new.append(word)

    output =  " ".join(output_words_new)

    if output == aux1 + " " + d1:
        return "d1p1"
    if output == aux2 + " " + d1:
        return "d1p2"
    if output == aux1 + " " + d2:
        return "d2p1"
    if output == aux2 + " " + d2:
        return "d2p2"
    if output == aux1 + " " + dn:
        return "dnp1"
    if output == aux2 + " " + dn:
        return "dnp2"
    if output.split()[0] in auxes:
        if output.split()[1:] == d1:
            return "d1po"
        if output.split()[1:] == d2:
            return "d2po"
        if output.split()[1:] == dn:
            return "dnpo"

    return "other"

# Given an auxiliary, return whether it is singular or plural
def number_aux(aux_word):
    if aux_word == "do" or aux_word == "don't":
        return "PL"
    return "SG"
    

# Determine whether a sentence contains exactly two auxiliaries
# which must agree in number (i.e., 2 singular auxiliaries or 
# 2 plural auxiliaries)
def two_agreeing_auxes(sent):
    aux_list = []
    for word in sent.split():
        if word in auxes:
            aux_list.append(word)

    if len(aux_list) != 2:
        return False

    else:
        if number_aux(aux_list[0]) == number_aux(aux_list[1]) and aux_list[0] != aux_list[1]:
            return True
        else:
            return False

nouns_sg = ["newt", "orangutan", "peacock", "quail", "raven", "salamander", "tyrannosaurus", "unicorn", "vulture", "walrus", "xylophone", "yak", "zebra"]
nouns_pl = ["newts", "orangutans", "peacocks", "quails", "ravens", "salamanders", "tyrannosauruses", "unicorns", "vultures", "walruses", "xylophones", "yaks", "zebras"]

verbs_sg = ["giggles", "smiles", "sleeps", "swims", "waits", "moves", "changes", "reads", "eats", "entertains", "amuses", "high_fives", "applauds", "confuses", "admires", "accepts", "remembers", "comforts"]
verbs_pl = ["giggle", "smile", "sleep", "swim", "wait", "move", "change", "read", "eat", "entertain", "amuse", "high_five", "applaud", "confuse", "admire", "accept", "remember", "comfort"]

auxes_sg = ["does", "doesn't"]
auxes_pl = ["do", "don't"]

# Given an input past tense sentence, outputs
# what the present-tense version would be if
# verbs agreed with the most recent noun instead
# of with their subjects.
def tense_nearest(sent):
    new_words = []
    words = sent.split()

    tense_agr = "sg"
    for word in words:
        if word in nouns_sg:
            tense_agr = "sg"
            new_words.append(word)
        elif word in nouns_pl:
            tense_agr = "pl"
            new_words.append(word)
        elif word in verbs_sg:
            verb_ind = verbs_sg.index(word)
            if tense_agr == "sg":
                new_words.append(verbs_sg[verb_ind])
            else:
                new_words.append(verbs_pl[verb_ind])
        elif word in verbs_pl:
            verb_ind = verbs_pl.index(word)
            if tense_agr == "sg":
                new_words.append(verbs_sg[verb_ind])
            else:
                new_words.append(verbs_pl[verb_ind])
        else:
            new_words.append(word)

    return " ".join(new_words)

# Same as tense_nearest, but when the sentence has auxes
def tense_nearest_aux(sent):
    new_words = []
    words = sent.split()

    tense_agr = "sg"
    for word in words:
        if word in nouns_sg:
            tense_agr = "sg"
            new_words.append(word)
        elif word in nouns_pl:
            tense_agr = "pl"
            new_words.append(word)
        elif word in auxes_sg:
            aux_ind = auxes_sg.index(word)
            if tense_agr == "sg":
                new_words.append(auxes_sg[aux_ind])
            else:
                new_words.append(auxes_pl[aux_ind])
        elif word in auxes_pl:
            aux_ind = auxes_pl.index(word)
            if tense_agr == "sg":
                new_words.append(auxes_sg[aux_ind])
            else:
                new_words.append(auxes_pl[aux_ind])
        else:
            new_words.append(word)

    return " ".join(new_words)





# Converting a sentence to a list of part-of-speech tags
posDictTense = {}
fi = open("pos_tense.txt", "r")
for line in fi:
    parts = line.split("\t")
    posDictTense[parts[0].strip()] = parts[1].strip()
        
posDict = {}
fi = open("pos.txt", "r")
for line in fi:
    parts = line.split("\t")
    posDict[parts[0].strip()] = parts[1].strip()
        

def sent_to_pos(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        if word in posDict:
            pos_tags.append(posDict[word])
        else:
            pos_tags.append(posDictTense[word])

    return pos_tags

# Determines whether an output is correct except for the verbs
def right_besides_verbs(senta, sentb):
    pos_tags = sent_to_pos(senta)
    
    wordsa = senta.split()
    wordsb = sentb.split()
    
    if len(wordsb) != len(wordsa):
        return False

    all_good = True
    
    for index, word in enumerate(wordsa):
        if pos_tags[index] == "V":
            continue
       	 
        if word != wordsb[index]:
            all_good = False
            
    return all_good

# Does the sentence have the correct sequence of
# part of speech tags?
def right_pos(senta, sentb):
    pos_tags_a = sent_to_pos(senta)
    pos_tags_b = sent_to_pos(sentb)

    if pos_tags_a == pos_tags_b:
        return True
    else:
        return False

# Is the sentence correct except for the identity
# of the auxiliaries?
def right_besides_auxes(senta, sentb):
    pos_tags = sent_to_pos(senta)

    wordsa = senta.split()
    wordsb = sentb.split()

    if len(wordsb) != len(wordsa):
        return False

    all_good = True

    for index, word in enumerate(wordsa):
        if pos_tags[index] == "A":
            continue

        if word != wordsb[index]:
            all_good = False

    return all_good





# Determines whether the main verb of the sentence is correct
def main_right_tense(senta, sentb):
    
    if not right_pos(senta, sentb):
        return False
    
    wordsa = senta.split()
    wordsb = sentb.split()
    
    pos_tags = sent_to_pos(senta)
    
    if pos_tags[2] == "R":
        seen_v = 0
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                if seen_v:
                    ind_v = index
                    break
                else:
                    seen_v = 1
                    
                    
    else:
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                ind_v = index
                break
                
    verba = wordsa[ind_v]
    verbb = wordsb[ind_v]
    
    return verbb == verba


# Determines whether the auxiliary of the main verb of the sentence is correct
def main_right_tense_aux(senta, sentb):

    if not right_pos(senta, sentb):
        return False

    wordsa = senta.split()
    wordsb = sentb.split()

    pos_tags = sent_to_pos(senta)

    if pos_tags[2] == "R":
        seen_a = 0
        for index, tag in enumerate(pos_tags):
            if tag == "A":
                if seen_a:
                    ind_a = index
                    break
                else:
                    seen_a = 1


    else:
        for index, tag in enumerate(pos_tags):
            if tag == "A":
                ind_a = index
                break

    auxa = wordsa[ind_a]
    auxb = wordsb[ind_a]

    return auxb == auxa



# Determines whether the main verb of the sentence is the one predicted by agree-recent
def main_linear_tense(senta, sentb):

    if not right_pos(senta, sentb): 
        return False

    wordsa = senta.split()
    wordsb = sentb.split()

    pos_tags = sent_to_pos(senta)

    if pos_tags[2] == "R":
        seen_v = 0
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                if seen_v:
                    ind_v = index
                    break
                else:
                    seen_v = 1


    else:
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                ind_v = index
                break

    verba = wordsa[ind_v]
    verbb = wordsb[ind_v]

    if verbb + "s" == verba:
       return True
    if verbb == verba + "s":
       return True
    return False

# Determines whether the auxiliary of the main verb of the sentence is the one predicted by agree-recent
def main_linear_tense_aux(senta, sentb):

    if not right_pos(senta, sentb):
        return False

    wordsa = senta.split()
    wordsb = sentb.split()

    pos_tags = sent_to_pos(senta)

    if pos_tags[2] == "R":
        seen_a = 0
        for index, tag in enumerate(pos_tags):
            if tag == "A":
                if seen_a:
                    ind_a = index
                    break
                else:
                    seen_a = 1


    else:
        for index, tag in enumerate(pos_tags):
            if tag == "A":
                ind_a = index
                break

    auxa = wordsa[ind_a]
    auxb = wordsb[ind_a]

    if auxa == "do" and auxb == "does":
        return True
    if auxa == "does" and auxb == "do":
        return True
    if auxa == "don't" and auxb == "doesn't":
        return True
    if auxa == "doesn't" and auxb == "don't":
        return True
    return False


    return auxb == auxa



# Determines whether the main verb of the sentence has the number predicted by agree-recent
def main_wrongnum_tense(senta, sentb):

    if not right_pos(senta, sentb): 
        return False

    wordsa = senta.split()
    wordsb = sentb.split()

    pos_tags = sent_to_pos(senta)

    if pos_tags[2] == "R":
        seen_v = 0
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                if seen_v:
                    ind_v = index
                    break
                else:
                    seen_v = 1


    else:
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                ind_v = index
                break

    verba = wordsa[ind_v]
    verbb = wordsb[ind_v]

    if verbb[-1] == "s" and verba[-1] != "s":
       return True
    if verbb[-1] != "s" and verba[-1] == "s":
       return True
    return False

# Determines whether the auxiliary of the main verb of the sentence has the number predicted by agree-recent
def main_wrongnum_tense_aux(senta, sentb):

    if not right_pos(senta, sentb):
        return False

    wordsa = senta.split()
    wordsb = sentb.split()

    pos_tags = sent_to_pos(senta)

    if pos_tags[2] == "R":
        seen_a = 0
        for index, tag in enumerate(pos_tags):
            if tag == "A":
                if seen_a:
                    ind_a = index
                    break
                else:
                    seen_a = 1


    else:
        for index, tag in enumerate(pos_tags):
            if tag == "A":
                ind_a = index
                break

    auxa = wordsa[ind_a]
    auxb = wordsb[ind_a]

    if auxa == "do" and auxb == "does":
        return True
    if auxa == "do" and auxb == "doesn't":
        return True
    if auxa == "don't" and auxb == "does":
        return True
    if auxa == "don't" and auxb == "doesn't":
        return True
    if auxa == "does" and auxb == "do":
        return True
    if auxa == "does" and auxb == "don't":
        return True
    if auxa == "doesn't" and auxb == "do":
        return True
    if auxa == "doesn't" and auxb == "don't":
        return True

    return False


# Determines whether the main verb of the sentence has the correct number
def main_rightnum_tense(senta, sentb):

    if not right_pos(senta, sentb): 
        return False

    wordsa = senta.split()
    wordsb = sentb.split()

    pos_tags = sent_to_pos(senta)

    if pos_tags[2] == "R":
        seen_v = 0
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                if seen_v:
                    ind_v = index
                    break
                else:
                    seen_v = 1


    else:
        for index, tag in enumerate(pos_tags):
            if tag == "V":
                ind_v = index
                break

    verba = wordsa[ind_v]
    verbb = wordsb[ind_v]

    if verba[-1] == "s" and verbb[-1] == "s":
        return True
    if verba[-1] != "s" and verbb[-1] != "s":
        return True
    return False

# Determines whether the auxiliary of the main verb of the sentence has the correct number
def main_rightnum_tense_aux(senta, sentb):

    if not right_pos(senta, sentb):
        return False

    wordsa = senta.split()
    wordsb = sentb.split()

    pos_tags = sent_to_pos(senta)

    if pos_tags[2] == "R":
        seen_a = 0
        for index, tag in enumerate(pos_tags):
            if tag == "A":
                if seen_a:
                    ind_a = index
                    break
                else:
                    seen_a = 1


    else:
        for index, tag in enumerate(pos_tags):
            if tag == "A":
                ind_a = index
                break

    auxa = wordsa[ind_a]
    auxb = wordsb[ind_a]

    if auxa == "do" and auxb == "do":
        return True
    if auxa == "do" and auxb == "don't":
        return True
    if auxa == "don't" and auxb == "do":
        return True
    if auxa == "don't" and auxb == "don't":
        return True
    if auxa == "does" and auxb == "does":
        return True
    if auxa == "does" and auxb == "doesn't":
        return True
    if auxa == "doesn't" and auxb == "does":
        return True
    if auxa == "doesn't" and auxb == "doesn't":
        return True

    return False



