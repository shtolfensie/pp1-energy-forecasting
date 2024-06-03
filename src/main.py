import argparse
from datetime import datetime
from pathlib import Path
from construct_datasets import DataSet
import models.cnn as cnn
import models.tcn_lstm as lstm
import evaluation
import preprocess_dataset as dat
import torch



def save_checkpoint(name:str, checkpoint_dir: Path, model, architecture: str, features: list[str]):
    (checkpoint_dir / name).mkdir(exist_ok=True)
    model.save(checkpoint_dir / name / f"{name}.keras")


def save_eval(res, name:str, checkpoint_dir: Path, architecture: str, features: list[str], epochs: int, show: bool = False):
    (checkpoint_dir / name).mkdir(exist_ok=True)

    rmse, mape, nrmse = evaluation.average_metrics(res)

    eval_file = checkpoint_dir / name / f"{name}-eval.txt"

    eval_lines = [
        f"name: {name}\n",
        f"epochs: {epochs}\n",
        f"architecture: {architecture}\n",
        f"features: {features}\n",
        f"-------------\n",
        f"rmse: {rmse}\n",
        f"mape: {mape}\n",
        f"nrmse: {nrmse}\n",
    ]


    with open(eval_file, "w+") as f:
        f.writelines(eval_lines)

    if show:
        print("".join(eval_lines))


def run_experiment(dataset: str, features: list[str], architecture: str, generalize_test: bool, checkpoint_dir: Path, epochs: int):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')



    name = f"{architecture}-{dataset}-[{'_'.join(features)}]-{datetime.now()}"

    if architecture == "tcn":
        model = cnn.create_tcn(len(features))
        model.to(device)
    elif architecture == "tcn-lstm":
        model = lstm.create_tcn_lstm(len(features))
        model.to(device)
    else:
        raise Exception("Unknown network architecture:", architecture)


    univariate = len(features) == 1
    

    if univariate:
        data_func = dat.nist_univariate if dataset=="nist" else dat.fr_univariate
        X_train, y_train, X_test, y_test = data_func()

        X_train = torch.tensor(X_train).to(device)
        y_train = torch.tensor(y_train).to(device)


        model.fit(X_train, y_train, epochs=epochs)

        save_checkpoint(name=name, checkpoint_dir=checkpoint_dir, model=model, architecture=architecture, features=features)

        print("Running evaluation...")
        y_hat = model.predict(X_test)
        eval_res = evaluation.eval_forecast(y=y_test, y_hat=y_hat, arch=architecture)

    else:
        if dataset != "nist":
            raise Exception("Training multivariate models is currently implemented only for the NIST dataset")

        X_train, X_train_exo_np, y_train, X_test, X_test_exo, y_test = dat.nist_multivariate(path=None, exo_list=features)

        X_train = torch.tensor(X_train).to(device)
        y_train = torch.tensor(y_train).to(device)

        X_train_exo = []
        for exo in X_train_exo_np:
            X_train_exo.append(torch.tensor(exo).to(device))


        model.fit([X_train, *X_train_exo], y_train, epochs=epochs)

        save_checkpoint(name=name, checkpoint_dir=checkpoint_dir, model=model, architecture=architecture, features=features)

        print("Running evaluation...")
        y_hat = model.predict([X_test, *X_test_exo])
        eval_res = evaluation.eval_forecast(y=y_test, y_hat=y_hat)


    save_eval(eval_res, name=name, checkpoint_dir=checkpoint_dir, architecture=architecture, features=features, epochs=epochs, show=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-f", "--features", choices=["history", "temperature", "humidity", "wind"], required=True, action="append")
    parser.add_argument("--network", choices=["tcn", "tcn-lstm"], required=True)
    parser.add_argument("--generalize_test", action="store_true")
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("-d", "--dataset", default="nist", choices=["nist", "ihepc"])
    parser.add_argument("-e", "--epochs", default=1000, type=int)

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir is not None else Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    run_experiment(dataset=args.dataset, features=args.features, architecture=args.network, generalize_test=args.generalize_test, checkpoint_dir=checkpoint_dir, epochs=args.epochs)
    

