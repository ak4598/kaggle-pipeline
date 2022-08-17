import os
import yaml
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Kaggle pipeline')
    parser.add_argument('--cfg', type=str,
                        help='cfg file path')
    parser.add_argument('-m', '--model', type=str, help='model name')
    return parser.parse_args()


def main():
    args = parse_args()

    CFG_FILE = args.cfg

    with open(CFG_FILE) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    dataset = getattr(__import__("data"), cfg["modules"]["data"])
    model = getattr(__import__("models"), cfg["modules"]["model"])

    d = dataset(cfg)

    if cfg["dataset"]["prepare"]:
        d.clean()
        d.split()

    X_train, Y_train, X_test, Y_test = d.load()

    m = model(cfg)
    m.build()
    best_model, best_params = m.train(X_train, Y_train)
    final_score = m.eval(X_test, Y_test)

    if cfg["model"]["train"]["save"]:
        d.save(best_model, best_params, final_score)


if __name__ == '__main__':
    main()
