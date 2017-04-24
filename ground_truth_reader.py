import json

def build_line_meta(file):
    gt = {"lines": []}
    with open(file) as f:
        ground_truth = json.loads(f.read())

        for line in ground_truth["lines"]:
            pass

    return ground_truth
