import argparse
from datasets import load_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred")
    parser.add_argument("--label")
    args = parser.parse_args()
    metric = load_metric("cer")
    with open(args.pred, 'r') as pred, open(args.label, 'r') as label:
        preds = pred.readlines()
        labels = label.readlines()
        preds = [line.strip() for line in preds]
        labels = [line.strip() for line in labels]
        print("prediction: {} labels: {} cer: {:.3f}".format(args.pred, args.label, metric.compute(predictions=preds, references=labels)))
