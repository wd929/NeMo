import argparse
import os
import csv

# Parsing arguments
parser = argparse.ArgumentParser(
    description='Joint intent slot filling system with pretrained BERT')
parser.add_argument("--data_dir", default='data/mturk', type=str)
parser.add_argument("--dataset_name", default='mturk', type=str)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--do_lower_case", action='store_false')
parser.add_argument("--shuffle_data", action='store_false')

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'

# with open(args.data_dir, 'r') as f:
#     utterances = f.readlines()

utterances = []

with open(args.data_dir, 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        utterances.append(row)
 #            print(row)
            # print(row[0])
            # print(row[1])


def analyze_inter_annotator_agreement(num_annotators, utterances):
    print("Agreement here")
    print(f'You had {num_annotators} annotators')

    # # TO DO
    # print("Assuming all the utterances are present contiguously -- Describe assumed format here:")
    size = num_annotators

    classes = []

    print(utterances[1:12])

    # TO DO - Make generalizable to n number of annotators.
    agreedall = []
    agreedtwo = []
    disagreedall = []



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

    print(len(agreedall))
    print(len(agreedtwo) + len(agreedall))
    print(len(disagreedall))

    print("Full agreement")
    print(agreedall[:10])
    print("2 / 3 agreement")
    print(agreedtwo[:10])
    print("No agreement")
    print(disagreedall[:10])



analyze_inter_annotator_agreement(3, utterances)