import argparse
import os
import csv
import json
import random

# Parsing arguments
parser = argparse.ArgumentParser(
    description='Joint intent slot filling system with pretrained BERT')
parser.add_argument("--data_dir", default='data/mturk', type=str)
parser.add_argument("--classification_data", default='data/mturk/classification.csv', type=str)
parser.add_argument("--annotation_data", default='data/mturk/annotation.manifest', type=str)
parser.add_argument("--num_annotators", default=3, type=str)
parser.add_argument("--dataset_name", default='mturk', type=str)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--do_lower_case", action='store_false')
parser.add_argument("--shuffle_data", action='store_false')

args = parser.parse_args()

if not os.path.exists(args.classification_data):
    raise ValueError(f'Data not found at {args.classification_data}')

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'

# TO DO  Change data_dir to classification data
utterances = []

outfold = f'{args.data_dir}/mturk/nemo-processed'
os.makedirs(outfold, exist_ok=True)

with open(args.classification_data, 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        utterances.append(row)

def analyze_inter_annotator_agreement(num_annotators, utterances, use_two_agreed=False):
    print("Agreement here")
    print(f'You had {num_annotators} annotators')

    # # TO DO
    # print("Assuming all the utterances are present contiguously 
    # Describe assumed format here:")
    size = num_annotators

    classes = []
    classDict = {}

    # TO DO - Make generalizable to n number of annotators.
    agreedall = []
    agreedtwo = []
    disagreedall = []

    intent_names = {}
    intent_count = 0



    for i, utterance in enumerate(utterances[1:]):
        if size > 0:
            if utterance[1] not in classes:
                classes.append(utterance[1])
                classDict[utterance[1]] = 1
            else:
                classDict[utterance[1]] = classDict.get(utterance[1]) + 1
            size -= 1
        if size == 0:
            if len(classes) == 1:
                agreedall.append([utterance[0], classes[0]])
            elif len(classes) == 2:
                agreedtwo.append([utterance[0], classes[0], 
                        classes[1]])
            elif len(classes) == 3:
                disagreedall.append([utterance[0], classes[0], 
                        classes[1], classes[2]])
            else:
                raise ValueError(f'Number of annotators is more than {num_annotators}')

            # use_two_agreed # currently specific to 3 annotators only
            # TO DO : Generalize to n annotators
            if use_two_agreed and num_annotators - len(classes) + 1 == 2:
                if classDict[classes[0]] > classDict[classes[1]]:
                    agreedall.append([utterance[0], classes[0]])
                else:
                    agreedall.append([utterance[0], classes[1]])

            size = num_annotators
            classes = []
        if utterance[1] not in intent_names:
            intent_names[utterance[1]] = str(intent_count)
            intent_count += 1

    print(len(agreedall))
    print(len(agreedtwo)) # + len(agreedall))
    print(len(disagreedall))

    print("Full agreement")
    print(agreedall[:10])
    print("2 / 3 agreement")
    print(agreedtwo[:10])
    print("No agreement")
    print(disagreedall[:10])

    agreedallDict = {}

    agreedallDict.update(agreedall)

    # print(len(agreedallDict))
    # print(agreedallDict.get('How many point do I need before my first reward?'))

    return agreedall, agreedallDict, intent_names


def get_slot_labels(slot_annotations):

    slot_labels = json.loads(slot_annotations[0])

    all_labels = {}
    count = 0
    for label in slot_labels['retail-test']['annotations']['labels']:
        all_labels[label['label']] = str(count)
        count += 1
    all_labels['O'] = str(count)

    return all_labels


def process_intent_slot(slot_annotations, agreedallDict, intent_names):

    slot_tags = []
    inorder_utterances = []
    all_labels = get_slot_labels(slot_annotations)
    print(len(agreedallDict))


    for annotation in slot_annotations[0:]:
        an = json.loads(annotation)
        utterance = an['source']

        if utterance in agreedallDict:
            entities = {}
            
            # TO DO - Change the below name (retail-test) :
            for i, each_anno in enumerate(an['retail-test']['annotations']['entities']):
                entities[int(each_anno['startOffset'])] = i

            lastptr = 0
            slots = ""
            #sorting annotations by the start offset
            for i in sorted(entities.keys()):
                tags = an['retail-test']['annotations']['entities'][entities.get(i)]
                untagged_words = utterance[lastptr:tags['startOffset']] #utterance.substr(lastptr, tags['startOffset'])
                for word in untagged_words.split():
                    slots = ' '.join([slots, all_labels.get('O')])
                anno_words = utterance[tags['startOffset']:tags['endOffset']]
                for word in anno_words.split():
                    slots = ' '.join([slots, all_labels.get(tags['label'])])
                lastptr = tags['endOffset']

            untagged_words = utterance[lastptr:len(utterance)]
            for word in untagged_words.split():
                slots = ' '.join([slots, all_labels.get('O')])
            slots = f'{slots.strip()}\n'
            slot_tags.append(slots)
            intent_num = intent_names.get(agreedallDict.get(utterance))
            querytext = f'{utterance.strip()}\t{intent_num}\n'
            inorder_utterances.append(querytext)
    print(len(inorder_utterances))

    return all_labels, inorder_utterances, slot_tags

# The following works for the specified DialogFlow and Mturk output format
def partition_data(intent_queries, slot_tags, split=0.1):
    n = len(intent_queries)
    n_dev = int(n * split)
    dev_idx = set(random.sample(range(n), n_dev))
    dev_intents, dev_slots, train_intents, train_slots = [], [], [], []

    dev_intents.append('sentence\tlabel\n')
    train_intents.append('sentence\tlabel\n')

    for i, item in enumerate(intent_queries):
        if i in dev_idx:
            dev_intents.append(item)
            dev_slots.append(slot_tags[i])
        else:
            train_intents.append(item)
            train_slots.append(slot_tags[i])
    return train_intents, train_slots, dev_intents, dev_slots

# The following works for the specified DialogFlow and Mturk output format
def write_files(data, outfile):
    with open(f'{outfile}', 'w') as f:
        for item in data:
            item = f'{item.strip()}\n'
            f.write(item)

use_two_agreed = False

agreedall, agreedallDict, intent_names = analyze_inter_annotator_agreement(
    args.num_annotators, 
    utterances, 
    use_two_agreed)

with open(args.annotation_data, 'r') as f:
    slot_annotations = f.readlines()

slot_labels, intent_queries, slot_tags = process_intent_slot(
        slot_annotations, 
        agreedallDict, 
        intent_names)

assert len(slot_tags) == len(intent_queries)

# for i in range(len(slot_tags)):
#     print(intent_queries[i])
#     print(slot_tags[i])
#     print("--------------------------")


dev_split = 0.1

train_queries, train_slots, test_queries, test_slots = \
    partition_data(intent_queries, slot_tags, split=dev_split)

print(len(train_queries))
print(len(test_queries))
print(len(train_slots))
print(len(test_slots))


write_files(train_queries, f'{outfold}/train.tsv')
write_files(train_slots, f'{outfold}/train_slots.tsv')

write_files(test_queries, f'{outfold}/test.tsv')
write_files(test_slots, f'{outfold}/test_slots.tsv')

write_files(slot_labels, f'{outfold}/dict.slots.csv')
write_files(intent_names, f'{outfold}/dict.intents.csv')