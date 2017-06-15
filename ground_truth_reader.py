import json

def build_line_meta(file):
    with open(file) as f:
        ground_truth = json.loads(f.read())

    return ground_truth['lines']
