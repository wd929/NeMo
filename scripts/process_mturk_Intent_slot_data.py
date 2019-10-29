import argparse
import os
import csv
import json

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

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'

# TO DO  Change data_dir to classification data
utterances = []

with open(args.data_dir, 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        utterances.append(row)
def analyze_inter_annotator_agreement(num_annotators, utterances):
    print("Agreement here")
    print(f'You had {num_annotators} annotators')

    # # TO DO
    # print("Assuming all the utterances are present contiguously -- Describe assumed format here:")
    size = num_annotators

    classes = []

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
            size -= 1
        if size == 0:
            if len(classes) == 1:
                agreedall.append([utterance[0], classes[0]])
            elif len(classes) == 2:
                agreedtwo.append([utterance[0], classes[0], classes[1]])
            elif len(classes) == 3:
                disagreedall.append([utterance[0], classes[0], classes[1], classes[2]])
            else:
                raise ValueError(f'Number of annotators is more than {num_annotators}')
            size = num_annotators
            classes = []
        if utterance[1] not in intent_names:
        	intent_names[utterance[1]] = str(intent_count)
        	intent_count += 1

    print(len(agreedall))
    print(len(agreedtwo) + len(agreedall))
    print(len(disagreedall))

    print("Full agreement")
    print(agreedall[:10])
    print("2 / 3 agreement")
    print(agreedtwo[:10])
    print("No agreement")
    print(disagreedall[:10])

    return agreedall, intent_names


def process_slot_annotation(slot_annotations):

	slotdict = {}

	slot_labels = json.loads(slot_annotations[0])

	all_labels = {}
	count = 0
	for label in slot_labels['retail-test']['annotations']['labels']:
		# print(label['label'])
		all_labels[label['label']] = str(count)
		count += 1
	all_labels['O'] = str(count)


	for annotation in slot_annotations[160:190]:
		an = json.loads(annotation)
		utterance = an['source']

		entities = {}
		slot_tags = []
		# TO DO - Change the below name (retail-test) :
		for each in an['retail-test']['annotations']['entities']:
			entities[int(each['startOffset'])] = (
				each['label'], 
				utterance[each['startOffset']:each['endOffset']])
		# entities = sorted(entities.keys()) 
		for key in sorted(entities.keys()):
			sortedentity.
		print(entities)
		slotdict[utterance] = entities

	return all_labels, slotdict


def get_intents_and_slots(agreedall, intent_names):

    intent_queries = []

    for query in agreedall:
    	intent_num = intent_names.get(query[1])
    	querytext = f'{query[0].strip()}\t{intent_num}\n'
    	intent_queries.append(querytext)

    return intent_queries


agreedall, intent_names = analyze_inter_annotator_agreement(args.num_annotators, utterances)

intent_queries = get_intents_and_slots(agreedall, intent_names)


with open(args.annotation_data, 'r') as f:
    slot_annotations = f.readlines()

slot_labels, annoDict = process_slot_annotation(slot_annotations)

print(annoDict)

# print(slot_labels)

# train_queries, train_slots, test_queries, test_slots = \
#     partition_df_data(intent_queries, slot_tags, split=dev_split)