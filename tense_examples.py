import sys

nouns_sg = ["newt", "orangutan", "peacock", "quail", "raven", "salamander", "tyrannosaurus", "unicorn", "vulture", "walrus", "xylophone", "yak", "zebra"]
nouns_pl = ["newts", "orangutans", "peacocks", "quails", "ravens", "salamanders", "tyrannosauruses", "unicorns", "vultures", "walruses", "xylophones", "yaks", "zebras"]

verbs_sg = ["giggles", "smiles", "sleeps", "swims", "waits", "moves", "changes", "reads", "eats", "entertains", "amuses", "high_fives", "applauds", "confuses", "admires", "accepts", "remembers", "comforts"]
verbs_pl = ["giggle", "smile", "sleep", "swim", "wait", "move", "change", "read", "eat", "entertain", "amuse", "high_five", "applaud", "confuse", "admire", "accept", "remember", "comfort"]

verbs_t = ["entertains", "amuses", "high_fives", "applauds", "confuses", "admires", "accepts", "remembers", "comforts", "entertain", "amuse", "high_five", "applaud", "confuse", "admire", "accept", "remember", "comfort", "entertained", "amused", "high_fived", "applauded", "confused", "admired", "accepted", "remembered", "comforted"]
verbs_i = ["giggles", "smiles", "sleeps", "swims", "waits", "moves", "changes", "reads", "eats", "giggle", "smile", "sleep", "swim", "wait", "move", "change", "read", "eat", "giggled", "smiled", "slept", "swam", "waited", "moved", "changed", "read", "ate"]

nouns = nouns_sg + nouns_pl
verbs = verbs_t + verbs_i

preps = ["around", "near", "with", "upon", "by", "behind", "above", "below"]
rels = ["who", "that"]

posDictTense = {}
fi = open("pos_tense.txt", "r")
for line in fi:
        parts = line.split("\t")
        posDictTense[parts[0].strip()] = parts[1].strip()

def sent_to_pos(sent):
    words = sent.split()

    pos_tags = []

    for word in words:
        pos_tags.append(posDictTense[word])

    return pos_tags


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


def main_right_tense(senta, sentb):

    if not right_besides_verbs(senta, sentb):
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



def sent_type(sent):
    words = sent.split()
    new_words = []
    for word in words:
        if word not in ["PAST", "past", "PRESENT", "present", ".", "?"]:
            new_words.append(word)

    words = new_words


    if words[2] in preps:
        first = "pp"
        first_fine = "pp"
        if words[5] in verbs_t:
            s_type = "t"
        else:
            s_type = "i"        
    elif words[2] in rels:
        first = "rc"
        if words[3] in verbs:
            if words[4] in verbs:
                first_fine = "srci"
                if words[4] in verbs_t:
                    s_type = "t"
                else:
                    s_type = "i"
            else:
                first_fine = "srct"
                if words[6] in verbs_t:
                    s_type = "t"
                else:
                    s_type = "i"
        else:
            first_fine = "orc"
            if words[6] in verbs_t:
                s_type = "t"
            else:
                s_type = "i"
    else:
        first = "none"
        first_fine = "none"
        if words[2] in verbs_t:
            s_type = "t"
        else:
            s_type = "i"

    if s_type == "i":
        return s_type + "_" + first, s_type + "_" + first_fine

    if words[-3] in preps:
        second = "pp"
        second_fine = "pp"
    elif words[-2] in rels:
        second = "rc"
        second_fine = "srci"
    elif words[-4] in rels:
        second = "rc"
        if words[-1] in nouns:
            second_fine = "srct"
        else:
            second_fine = "orc"
    else:
        second = "none"
        second_fine = "none"

    return s_type + "_" + first + "_" + second, s_type + "_" + first_fine + "_" + second_fine

#fi = open(sys.argv[1])
cat_dict = {}

counter = 0
input_category = sys.argv[1]

for i in range(10):
    counter = 0
    fi = open("tense_TREENew_0_0.001_256_" + str(i) + ".test_save", "r")

    for line in fi:
        if counter < 10000:
            counter += 1
            continue

        parts = line.strip().split("\t")
        category = sent_type(parts[1])
        if category[1] == input_category:
            print(parts[1])
            print(parts[2])
            print("")

        if category not in cat_dict:
            cat_dict[category] = [0,0,0]

        cat_dict[category][2] += 1
        if parts[1] == parts[2]:
            cat_dict[category][0] += 1
        if main_right_tense(parts[1], parts[2]):
            cat_dict[category][1] += 1


#    print(parts[1])

#    print(sent_type(parts[1]))


#for key in cat_dict:
#    print(key, cat_dict[key][0] * 1.0 / cat_dict[key][2], cat_dict[key][1])
#print("")
#for key in cat_dict:
#    print(key, cat_dict[key][1] * 1.0 / cat_dict[key][2], cat_dict[key][1])




