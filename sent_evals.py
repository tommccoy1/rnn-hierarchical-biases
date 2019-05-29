
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


        d1 = " ".join(words[:index1] + words[index1 + 1:-1]) # -1 added with removal of decl, quest
        d2 = " ".join(words[:index2] + words[index2 + 1:-1]) # -1 added with removal of decl, quest
        dn = " ".join(words[:-1]) # brackets added with removal of decl, quest
        if d1[-1] != ".":
            d1 = d1 + " ."
        if d2[-1] != ".":
            d2 = d2 + " ."
        if dn[-1] != ".":
            dn = dn + " ."

        #print(d1)
        #print(d2)
        #print(dn)
        output = output.replace("?", ".").replace("decl", "quest").replace("DECL", "quest").replace("QUEST", "quest")
        output_words = output.split()
        output_words_new = []

        for word in output_words:
            if word != "-q" and word != "-Q" and word != "+q" and word != "+Q" and word != "t":
                output_words_new.append(word)

        output =  " ".join(output_words_new)


        #print(output)
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


def number_aux(aux_word):
    if aux_word == "do" or aux_word == "don't":
        return "PL"
    return "SG"
    


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


		
posDictTense = {}
fi = open("pos_tense.txt", "r")
for line in fi:
        parts = line.split("\t")
        posDictTense[parts[0].strip()] = parts[1].strip()
        
posDict = {}
fi = open("pos.txt", "r") # MIGHT NEED TO CHANGE BACK
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
    
    

