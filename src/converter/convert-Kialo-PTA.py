from argparse import ArgumentParser
import pandas as pd
import json
import os


def process_premise_substance(data):
    premise_substance_data = []
    for _, entry in data.iterrows():
        premise_substance_data.append({
            'input': entry['premise'],
            'output': entry['prem_substance'] if 'prem_substance' in entry else 'N/A'
        })
    return premise_substance_data


def process_conclusion_substance(data):
    conclusion_substance_data = []
    for _, entry in data.iterrows():
        conclusion_substance_data.append({
            'input': entry['conclusion'],
            'output': entry['conc_substance'] if 'conc_substance' in entry else 'N/A'
        })
    return conclusion_substance_data


def process_argument_canonicalization(data):
    argument_canonicalization_data = []
    for _, entry in data.iterrows():
        argument_canonicalization_data.append({
            'input': {
                'premise': entry['premise'],
                'conclusion': entry['conclusion']
            },
            'output': entry['canonical_form'] if 'canonical_form' in entry else {}
        })
    return argument_canonicalization_data


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    data = pd.read_pickle('data/kialo-pta24.pkl')

    # premise_substance_data = process_premise_substance(data)
    # save_json(premise_substance_data, 'premise-substance-detection.json')
    #
    # conclusion_substance_data = process_conclusion_substance(data)
    # save_json(conclusion_substance_data, 'conclusion-substance-detection.json')

    argument_canonicalization_data = process_argument_canonicalization(data)
    save_json(argument_canonicalization_data, 'argument-canonicalization.json')

    print("Data processing and saving completed.")


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Process Kialo PTA dataset and save outputs.")
    args = arg_parser.parse_known_args()[0]

    main()
