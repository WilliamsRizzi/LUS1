from collections import Counter

import math
from subprocess import Popen
import os

print 'LUS1 willo'

#FOLDER OFFSET
DATA = '../data/'
SANDBOX = 'sandbox/'
RESULTS = '../result/'

#TRAIN DATA
train_data_in = DATA + 'NLSPARQL.train.data'
train_feat_in = DATA + 'NLSPARQL.train.feats.txt'

#TEST DATA
test_data_in = DATA + 'NLSPARQL.test.data'
test_feat_in = DATA + 'NLSPARQL.test.feats.txt'

#LOG DATA
dictionary_log = SANDBOX + 'dictionary.txt'
concepts_c_log = SANDBOX + 'concepts_c.txt'
concepts_log = SANDBOX + 'concepts.txt'
word2concept_log = SANDBOX + 'word2concept.txt'
word2concept_cutted_log = SANDBOX + 'word2concept_cutted.txt'
ngram_log = SANDBOX + 'ngram.txt'
sentences_log = SANDBOX + 'sentences.txt'
probability_log = SANDBOX + 'probabilities.txt'
lexicon_log = SANDBOX + 'lexicon.txt'
likelihood_log = SANDBOX + 'likelihood.descr'
training_sentences_log = SANDBOX + 'training_sentences.txt'
training_concepts_log = SANDBOX + 'training_concepts.txt'
training_lem_log = SANDBOX + 'training_lem.txt'
training_pos_log = SANDBOX + 'training_pos.txt'

#MODELS
likelihood_fst_model = SANDBOX + 'likelihood.fst'
lm_model = SANDBOX + 'lm.fst'
trained_model = SANDBOX + 'trained_model.fst'
test_sentence_model = SANDBOX + 'test_sentence_model.fst'
modeled_test_sentence_model = SANDBOX + 'modeled_sentence.fst'

#OUTPUT DATA
result_out = RESULTS + 'result.txt'
performances_out = RESULTS + 'performances.txt'

def import_data():
    """ Import the train data from the TRAIN_DATA_IN path variables
    :return: dictionary, concepts, word2concept,
        :rtype dictionary: list
        :rtype concepts: list
        :rtype word2concept: list
    """

    print '\n\timport train data'
    dictionary = []
    concepts = []
    word2concept = []
    dictionary += ['<s>']#start first sentence
    concepts += ['<s>']#start first sentence
    word2concept += [('<s>', '<s>')]
    for line in open(train_data_in, 'r'):
        line = line[:-1] #remove trailing \n
        values = line.split('\t')
        if len(values) >= 2:
            dictionary += [ str(values[0]) ]
            concepts += [ values[1] ]
            word2concept += [ (values[0], values[1]) ]
        else:
            dictionary += ['</s>']
            dictionary += [ '<s>' ]
            concepts += ['</s>']
            concepts += [ '<s>' ]
            word2concept += [('</s>', '</s>')]
            word2concept += [('<s>', '<s>')]
    dictionary += ['</s>']#end last sentence
    concepts += ['</s>']#end last sentence
    word2concept += [('</s>', '</s>')]

    f = open(training_sentences_log, 'w+')
    [ f.write('\n') if word in ['<s>', '</s>'] else f.write(' '+ word) for word in dictionary ]
    f.close()
    f = open(training_concepts_log, 'w+')
    [ f.write('\n') if word in ['<s>', '</s>'] else f.write(' '+ word) for word in concepts ]
    f.close()
    print '\t--done'
    return dictionary, concepts, word2concept

def import_lem_pos(dictionary, concepts, word2concept):
    """ Import the lem and pos data from the TRAIN_FEAT_IN path variables and integrates them in the retvals
    :param dictionary: all the tokens in the dataset,
    :param concepts: all the concepts in the dataset,
    :param word2concept: (word, concept) tuples.
        :type dictionary: list
        :type concepts: list
        :type word2concept: list
    :return: dictionary, concepts, word2concept,
        :rtype dictionary: list
        :rtype concepts: list
        :rtype word2concept: list
    """

    print '\n\timport lem and pos tagged data'
    lem_dictionary = []
    pos_dictionary = []
    lem_dictionary += [ '<s>' ]
    pos_dictionary += [ '<s>' ]
    for line in open(train_feat_in, 'r'):
        line = line[:-1]
        values = line.split('\t')
        if len(values) >= 3:
            lem_dictionary += [values[2]]
            pos_dictionary += [values[1]]
        else:
            lem_dictionary += [ '</s>' ]
            lem_dictionary += [ '<s>' ]
            pos_dictionary += [ '</s>' ]
            pos_dictionary += [ '<s>' ]
    lem_dictionary += [ '</s>' ]
    pos_dictionary += [ '</s>' ]

    f = open(training_lem_log, 'w+')
    [ f.write('\n') if word in ['<s>', '</s>'] else f.write(' '+ word) for word in lem_dictionary ]
    f.close()
    f = open(training_pos_log, 'w+')
    [ f.write('\n') if word in ['<s>', '</s>'] else f.write(' '+ word) for word in pos_dictionary ]
    f.close()

    annotated_word2concept = []
    miscellaneous = []

    for position in range(0, len(word2concept)):
        annotated_word2concept += [ (dictionary[position], word2concept[ position ][ 1 ]) ]
    dictionary += lem_dictionary #this SUMS lems in alg
    dictionary += miscellaneous #this SUMS lems in alg
    word2concept = annotated_word2concept #this subst lems in alg
    print '\t--done'
    return dictionary, concepts, word2concept

def remove_sentence_delimeters(dictionary, concepts, word2concept):
    """ Remove <s> and </s> from the threee input params
    :param dictionary: all the tokens in the dataset,
    :param concepts: all the concepts in the dataset,
    :param word2concept: (word, concept) tuples.
        :type dictionary: list
        :type concepts: list
        :type word2concept: list
    :return: dictionary, concepts, word2concept,
        :rtype dictionary: list
        :rtype concepts: list
        :rtype word2concept: list
    """

    print '\n\tremove sentence delimeters from dictionary and concepts'
    dictionary = [ word for word in dictionary if word not in ['<s>', '</s>'] ]
    concepts = [ word for word in concepts if word not in ['<s>', '</s>'] ]
    word2concept = [ word for word in word2concept if word[0] not in ['<s>', '</s>', '<s>.<s>', '</`s>.</s>'] ]
    print '\t--done'
    return dictionary, concepts, word2concept

def compute_dictionary_and_counts(dictionary_list):
    """ Init a Counter object with the dictionary list
    :param dictionary_list: all the tokens in the dataset,
        :type dictionary_list: list
    :return: dictionary
        :rtype dictionary: Counter
    """

    print '\tcreate dictionary and counts'
    dictionary = Counter(dictionary_list)
    dictionary['<unk>'] = 0 #oov
    f = open(dictionary_log, 'w+')
    [ f.write("{0}\t{1}\n".format(word, count)) for word, count in dictionary.most_common() ]
    f.close()
    print '\t--done'
    return dictionary

def compute_concept_counts_wout_iob(concepts):
    """ Init a Counter object with the concepts list without keeping into account the I-B annotation associated with it
    :param concepts: all the concepts in the dataset,
        :type concepts: list
    :return: concepts_c
        :rtype concepts_c: Counter
    """

    print '\n\tcounting concepts without IOB'
    concepts_c = Counter([ concept[2:] if concept != 'O' else concept for concept in concepts ]) #maybe i shold try to trim them on the I and B??
    f = open(concepts_c_log, 'w+')
    [ f.write("{0}\t{1}\n".format(word, count)) for word, count in concepts_c.most_common() ]
    f.close()
    print '\t--done'
    return concepts_c

def compute_concept_counts_w_iob(concepts_list):
    """ Init a Counter object with the concepts list keeping into account the I-B annotation associated with it
    :param concepts_list: all the concepts in the dataset,
        :type concepts_list: list
    :return: concepts
        :rtype concepts: Counter
    """

    print '\n\tcounting concepts'
    concepts = Counter(concepts_list) #maybe i shold try to trim them on the I and B??
    f = open(concepts_log, 'w+')
    [ f.write("{0}\t{1}\n".format(word, count)) for word, count in concepts.most_common() ]
    f.close()
    print '\t--done'
    return concepts

def write_lexicon(dictionary, concepts):
    """ Writes out the lexicon on the LEXICON_LOG file
    :param dictionary: all the tokens in the dataset,
    :param concepts: all the concepts in the dataset,
        :type dictionary: Counter
        :type concepts: Counter
    """

    print '\n\twrite lexicon'
    index = 0
    f = open(lexicon_log, 'w+')
    f.write('<eps>\t' + str(index) + '\n')
    index += 1
    for word in dictionary.keys():
        if word not in [ '<s>', '</s>' ]:
            f.write(word + '\t' + str(index) + '\n')
            index += 1
    for word in concepts.keys():
        f.write(word + '\t' + str(index) + '\n')
        index += 1
    f.write('<unk>\t' + str(index) + '\n')
    f.close()
    print '\t--done'

def create_word2concept(word2concept):
    """ Init a Counter object with the word2concept list
    :param word2concept: all the tuple(word, concept) in the dataset,
        :type word2concept: Counter
    :return: word2concept
        :rtype word2concept: Counter
    """

    print '\n\tcreate word2concept'
    word2concept = Counter(word2concept)
    f = open(word2concept_log, 'w+')
    [ f.write("{0}\t{1}\n".format(word, count)) for word, count in word2concept.most_common() ]
    f.close()
    print '\t--done'
    return word2concept

def filter_data(word2concept, high_cut_threshold=None, low_cut_threshold=None):
    """ IF set the thresholds:
            Filters out the data that have a count that lies in between the two thresholds
        AND integrates the <unk> to the word2concept variable
        FINALLY writes out the result on the WORD2CONCEPT_CUTTED_LOG in a suitable format for the fst
    :param word2concept: all the tuple(word, concept) in the dataset,
    :param high_cut_threshold: max amout of time a tuple must occour to be removed
    :param low_cut_threshold: min amout of time a tuple must occour to be removed
        :type word2concept: Counter
        :type high_cut_threshold: int
        :type low_cut_threshold: int
    :return: word2concept_cutted
        :rtype word2concept_cutted: Counter
    """

    word2concept_cutted = {}

    if high_cut_threshold is not None and low_cut_threshold is not None : #cut threshold specified (THRESHOLD METHOD)
        print '\n\tcompute cutoff - high_cut_threshold = ' + str(high_cut_threshold) + \
                                  ' low_cut_threshold = ' + str(low_cut_threshold)
        cumulative_counter = 0
        for key, value in word2concept.most_common():
            if value <= high_cut_threshold and value > low_cut_threshold:
                cumulative_counter += value
            else:
                word2concept_cutted[key] = value
        for concept in concepts.keys():
            word2concept_cutted[tuple(['<unk>', concept])] = cumulative_counter / float(len(concepts))

    else: #NO cut threshold specified (BASELINE METHOD)
        print '\n\tcompute out of vocabulary probs'
        for key, value in word2concept.most_common():
            word2concept_cutted[ key ] = value
        for concept in concepts.keys():
            word2concept_cutted[ tuple([ '<unk>', concept ]) ] = (len(concepts) / 10) / float(len(concepts))

    f = open(word2concept_cutted_log, 'w+')
    [ f.write("{0}\t{1}\n".format(word, count)) for word, count in word2concept_cutted.items() ]
    f.close()
    print '\t--done'
    return word2concept_cutted

def compute_likelihoods(word2concept_cutted):
    """ Computes the likelihood of each concept in the WORD2CONCEPT_CUTTED variable
    :param word2concept_cutted: all the tuple(word, concept) in the dataset,
        :type word2concept_cutted: Counter
    :return: word2concept_cutted, concepts
        :rtype word2concept_cutted: Counter
        :rtype likelihood: dict
    """

    print '\n\tcompute likelihood'
    likelihood = {}
    for key, value in word2concept_cutted.items():
        if concepts[key[1]] == 0:
            pass
        likelihood[key] = value / float(concepts[key[1]])

    for key, value in likelihood.iteritems():
        likelihood[key] = - math.log(value)
    print '\t--done'
    return word2concept_cutted, likelihood

def compute_fst(word2concept_cutted, likelihood):
    """ Builds the fst of the single words-concept association
    :param word2concept_cutted: all the tuple(word, concept) in the dataset,
    :param likelihood: all the tuple(word, concept) likelihoods in the dataset,
        :type word2concept_cutted: Counter
        :type likelihood: dict
    """

    print '\n\tcompute fst'
    f = open(likelihood_log, 'w+')
    [ f.write('0\t0\t' + key[0] + '\t' + key[1] + '\t%.5f' % likelihood[key] + '\n') for key in word2concept_cutted.keys() ]
    f.write('0')
    f.close()
    os.system(
        'fstcompile --isymbols=' + lexicon_log + ' --osymbols=' + lexicon_log + ' ' + likelihood_log + ' | ' +
        'fstarcsort > ' + likelihood_fst_model
    )
    print '\t--done'


def compute_lm(ngram_order=2, smoothing_method='absolute'):
    """ Builds the fst of language model
    :param ngram_order: desired ngram order,
    :param smoothing_method: desired smoothing method choose between [ "absolute", "katz", "kneser_ney", "presmoothed", "unsmoothed", "witten_bell" ],
        :type ngram_order: int
        :type smoothing_method: str
    """

    print '\n\tcompute lm fst'
    os.system(
        'farcompilestrings --symbols=' + lexicon_log + " --keep_symbols=1 --unknown_symbol='<unk>' " + training_concepts_log + ' | ' +
        'ngramcount --order=' + str(ngram_order) + ' | ' +
        'ngrammake --method=' + smoothing_method + ' | ' +
        'fstarcsort >' + lm_model
    )
    print '\t--done'

def compose_final_model():
    """ Composes the final model and puts in the TRAINED_MODEL path
    """

    print '\n\tcompute model from lm and likelihoods'
    os.system('fstcompose ' + likelihood_fst_model + ' ' + lm_model + ' > ' + trained_model)
    print '\t--done'

def test_model():
    """ Tests the model on the test set specified byt the TEST_DATA_IN and TEST_FEAT_IN and then puts the result in
    the RESULT_FILE
    """

    print '\n\ttest model'
    test_file = open(test_data_in, 'r')
    test_feat_file = open(test_feat_in, 'r')
    result_file = open(result_out, 'w+')
    sentences = []
    sentence = []
    concepts_all = []
    concepts = []
    for line in test_file:
        if line != '\n':
            line = line.split()
            sentence += [ line[0] ]
            concepts += [ line[1] ]
        else :
            sentences += [ sentence ]
            sentence = []
            concepts_all += [ concepts ]
            concepts = []

    pos_all = []
    pos = []
    lemmas = [ ]
    lemma = [ ]
    for line in test_feat_file :
        if line != '\n' :
            line = line.split()
            pos += [ line[1] ]
            lemma += [ line[2] ]
        else :
            pos_all += [ pos ]
            pos = []
            lemmas += [ lemma ]
            lemma = []
    test_file.close()
    test_feat_file.close()

    for position in range(len(sentences)):
        sentence = sentences[position]
        concepts = concepts_all[position]

        input_sentence = ''

        for position in range(len(sentence)):
            word = sentence[position]
            input_sentence += word if word in dictionary.keys() else "'<unk>'"
            input_sentence += ' '

        os.system(
            'echo "' + input_sentence + '" | ' +
            'farcompilestrings --symbols=' + lexicon_log + " --unknown_symbol='<unk>' --generate_keys=1 --keep_symbols | " +
            "farextract --filename_suffix='.fst' ; " +
            'mv 1.fst ' + test_sentence_model
        )
        os.system(
            'fstcompose ' + test_sentence_model + ' ' + trained_model + ' | ' +
            'fstshortestpath | ' +
            'fstrmepsilon | ' +
            'fsttopsort | ' +
            'fstprint --isymbols=' + lexicon_log + ' --osymbols=' + lexicon_log + ' > ' + modeled_test_sentence_model
        )
        sentence_model = open(modeled_test_sentence_model, 'r')
        index = 0
        input_sentence = input_sentence.split()
        for state in sentence_model:
            state = state.split()
            if len(state) == 5 and len(input_sentence) > index or len(concepts) > index :
                result_file.write(input_sentence[index] + ' ' + concepts[index] + ' ' + state[3] + '\n')
                index += 1
        result_file.write('\n')
        sentence_model.close()

    result_file.close() #write out
    print '\t--done'

def evaluate_model(ngram_order, smoothing_method, high_cut_threshold, low_cut_threshold):
    """ Evaluates the model using the ./conlleval.pl script that needs to be in the execution folder and then writes
    the result in the file specified by the PERFORMANCES_OUT varible
    :param ngram_order: desired ngram order,
    :param smoothing_method: desired smoothing method choose between [ "absolute", "katz", "kneser_ney", "presmoothed", "unsmoothed", "witten_bell" ],
    :param high_cut_threshold: max amout of time a tuple must occour to be removed
    :param low_cut_threshold: min amout of time a tuple must occour to be removed
        :type ngram_order: int
        :type smoothing_method: str
        :type high_cut_threshold: int
        :type low_cut_threshold: int
    """

    print '\n\tcompute evaluation metrics'
    performances_file = open(performances_out, 'a+')
    performances_file.write('\n\tSMOOTHING_METHOD ' + smoothing_method + '\tNGRAM_ORDER ' + str(ngram_order) + '\tLOW_CUT_THRESHOLD ' + str(low_cut_threshold) + '\tHIGH_CUT_THRESHOLD ' + str(high_cut_threshold) + '\n')
    performances_file.close()
    process = Popen(
        './conlleval.pl < ' + result_out + ' >> ' + performances_out,
        shell=True
    )
    process.communicate()
    print '\t--done'

if __name__ == "__main__" :

    #STEP 0: set parameters
    ngram_order = 2
    smoothing_method = 'absolute'
    high_cut_threshold = 5 #set me to None to run BASELINE
    low_cut_threshold = 2 #set me to None to run BASELINE

    #STEP 1: preprocess data
    dictionary, concepts, word2concept = import_data()
    dictionary, concepts, word2concept = import_lem_pos(dictionary, concepts, word2concept)
    dictionary, concepts, word2concept = remove_sentence_delimeters(dictionary, concepts, word2concept)

    #STEP 2: compute counts
    dictionary = compute_dictionary_and_counts(dictionary)
    concepts_c = compute_concept_counts_wout_iob(concepts)
    concepts = compute_concept_counts_w_iob(concepts)

    #STEP 3: write out dictionary
    write_lexicon(dictionary, concepts)

    #STEP 4: compute likelihoods
    word2concept = create_word2concept(word2concept)
    word2concept_cutted = filter_data(word2concept, high_cut_threshold, low_cut_threshold)
    word2concept_cutted, likelihood = compute_likelihoods(word2concept_cutted)

    #STEP 5: build model
    compute_fst(word2concept_cutted, likelihood)
    compute_lm(ngram_order, smoothing_method)
    compose_final_model()

    #STEP 6: test model
    test_model()
    evaluate_model(ngram_order, smoothing_method, high_cut_threshold, low_cut_threshold)

    #OPTIONAL STEP 7: cat out model performances
    os.system('cat ' + performances_out)

    #OPTIONAL STEP 8    : clean sandbox
    os.system('rm ' + SANDBOX + '*')

print '\n--DONE'
