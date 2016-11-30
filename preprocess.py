import os
import re
import numpy as np


# ############################# Function used from pre-processing ####################
# Read data from train and test data from Data directory 
def read_task_data(data_dir, data_id, only_supporting=False):
   
    # Make sure read data from available directory
    assert data_id > 0 and not 21 <= data_id

    # Reading Train and Test data from given directory    
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    story = 'qa{}_'.format(data_id)
    train_file = [f for f in files if story in f and 'train' in f][0]
    test_file = [f for f in files if story in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data


# Split sentences by space and get each word as token
def tokenize(sent):   
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


# Parsing story 
def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            # supporting - Index of sentence which has answer of question
            # e.g. Question has 'Sam', then consider only sentences which is related to 'Sam'
            question, answer, supporting = line.split('\t')
            question = tokenize(question)
            answer = [answer]
            # sub_story - contains only supporting sentences
            sub_story = None
            
            # Question in the story
            if question[-1] == "?":
                question = question[:-1]

            # Consider only supporting sentences
            if only_supporting:                
                supporting = map(int, supporting.split())
                sub_story = [story[i - 1] for i in supporting]
            else:                
                sub_story = [x for x in story if x]
            data.append((sub_story, question, answer))
            story.append('')
       
        else:
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)

    return data


# Parse story based on supporting sentences to question
def get_stories(f, only_supporting=False):
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


# Converted data into equivalent vector representation based word dictionary
def convert_to_vector(data, word_idx, sentence_size, memory_size):
    stories = []
    questions = []
    answers = []
    for story, question, answer in data:
        # ss - , ls - , lq - , q - , y -
        story_sentence = []
        for i, sentence in enumerate(story, 1):
            sentence_len = max(0, sentence_size - len(sentence))
            story_sentence.append([word_idx[w] for w in sentence] + [0] * sentence_len)

        story_sentence = story_sentence[::-1][:memory_size][::-1]
        memory_len = max(0, memory_size - len(story_sentence))
        for _ in range(memory_len):
            story_sentence.append([0] * sentence_size)

        question_len = max(0, sentence_size - len(question))
        ques = [word_idx[w] for w in question] + [0] * question_len

        y = np.zeros(len(word_idx) + 1) 
        for a in answer:
            y[word_idx[a]] = 1

        stories.append(story_sentence)
        questions.append(ques)
        answers.append(y)
    return np.array(stories), np.array(questions), np.array(answers)
