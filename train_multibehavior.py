from tqdm import tqdm
from time import time
import gc
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from model.simulator import *
from reader import *
import os
import utils
from sklearn.metrics import roc_auc_score
import json


def do_eval(model, reader, args, phase="val"):
    if phase not in ("val", "test"):
        raise ValueError(f"Unsupported evaluation phase: {phase}")

    previous_phase = reader.phase
    was_training = model.training
    reader.set_phase(phase)
    model.eval()
    eval_batch_size = args.val_batch_size if phase == "val" else args.test_batch_size
    eval_loader = DataLoader(reader, batch_size=eval_batch_size,
                             shuffle=False, pin_memory=False,
                             num_workers=reader.n_worker)
    eval_report = {'loss': [], 'auc': {}}
    Y_dict = {f: [] for f in model.feedback_types}
    P_dict = {f: [] for f in model.feedback_types}
    pbar = tqdm(total=len(reader))
    with torch.no_grad():
        for batch_data in eval_loader:
            wrapped_batch = utils.wrap_batch(batch_data, device=args.device)
            out_dict = model.do_forward_and_loss(wrapped_batch)
            loss = out_dict['loss']
            eval_report['loss'].append(loss.item())
            for j, f in enumerate(model.feedback_types):
                Y_dict[f].append(wrapped_batch[f].view(-1).detach().cpu().numpy())
                P_dict[f].append(out_dict['preds'][:, :, j].view(-1).detach().cpu().numpy())
            pbar.update(wrapped_batch['user_id'].shape[0])
    pbar.close()

    if not eval_report['loss']:
        reader.set_phase(previous_phase)
        model.train(was_training)
        raise ValueError(f"The {phase} split is empty; evaluation cannot be performed")

    eval_report['loss'] = (np.mean(eval_report['loss']),
                           np.min(eval_report['loss']),
                           np.max(eval_report['loss']))
    for f in model.feedback_types:
        eval_report['auc'][f] = roc_auc_score(np.concatenate(Y_dict[f]),
                                              np.concatenate(P_dict[f]))
    eval_report['mean_auc'] = float(np.mean(list(eval_report['auc'].values())))
    reader.set_phase(previous_phase)
    model.train(was_training)
    return eval_report


def calibrate_item_catalog(model, reader, args):
    """Give each catalog item equal representation-training opportunity.

    Interaction batches necessarily revisit popular items more often. This
    short, deterministic stage optimizes the same popularity-axis objective on
    the complete catalog while anchoring encodings to their pre-stage values.
    Only item-tower parameters and the auxiliary direction are updated.
    """
    if args.popularity_catalog_steps <= 0:
        return None
    if args.popularity_loss_coef <= 0.0:
        raise ValueError(
            "popularity_catalog_steps requires popularity_loss_coef > 0"
        )

    catalog = reader.get_item_catalog()
    item_ids = torch.as_tensor(
        catalog['item_id'], dtype=torch.long, device=args.device
    ).view(1, -1)
    labels = torch.as_tensor(
        catalog['item_type'], dtype=torch.float32, device=args.device
    ).view(1, -1)
    item_features = {
        key[3:]: torch.as_tensor(value, device=args.device).view(
            1, len(catalog['item_id']), -1
        )
        for key, value in catalog.items() if key.startswith('if_')
    }

    with torch.no_grad():
        reference, _ = model.get_item_encoding(item_ids, item_features, 1)
        reference = reference.detach()

    modules = [model.iIDEmb, model.itemEmbNorm, model.itemFeatureKernel]
    modules.extend(model.iFeatureEmb.values())
    parameters = [
        parameter for module in modules for parameter in module.parameters()
    ] + [model.popularityDirection]
    catalog_optimizer = torch.optim.Adam(
        parameters, lr=args.popularity_catalog_lr
    )
    report = None
    for step in range(1, args.popularity_catalog_steps + 1):
        catalog_optimizer.zero_grad()
        encoding, _ = model.get_item_encoding(item_ids, item_features, 1)
        popularity_loss = model.get_popularity_auxiliary_loss(
            encoding, labels
        )
        anchor_loss = torch.mean(torch.square(encoding - reference))
        loss = (
            args.popularity_loss_coef * popularity_loss
            + args.popularity_catalog_anchor_coef * anchor_loss
        )
        loss.backward()
        catalog_optimizer.step()
        report = {
            'step': step,
            'loss': float(loss.detach().cpu()),
            'popularity_loss': float(popularity_loss.detach().cpu()),
            'anchor_loss': float(anchor_loss.detach().cpu()),
        }
        if step == 1 or step % 50 == 0 or step == args.popularity_catalog_steps:
            print(f"Catalog calibration: {report}")
    return report


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--reader', type=str, required=True, help='Data reader class')
    init_parser.add_argument('--model', type=str, required=True, help='User response model class.')
    initial_args, _ = init_parser.parse_known_args()
    print(initial_args)
    modelClass = eval('{0}.{0}'.format(initial_args.model))
    readerClass = eval('{0}.{0}'.format(initial_args.reader))

    # control args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=128, help='validation batch size')
    parser.add_argument('--test_batch_size', type=int, default=128, help='test batch size')
    parser.add_argument('--save_with_val', action='store_true',
                        help='save the checkpoint with the best validation mean AUC')
    parser.add_argument('--early_stop_patience', type=int, default=3,
                        help='epochs without validation mean-AUC improvement before early stopping')
    parser.add_argument('--epoch', type=int, default=10, help='number of epoch')
    parser.add_argument('--cuda', type=int, default=-1, help='cuda device number; set to -1 (default) if using cpu')
    parser.add_argument(
        '--init_checkpoint', type=str, default='',
        help=(
            'optional checkpoint file used only to initialize model weights; '
            'the optimizer and validation selection start fresh'
        ),
    )
    parser.add_argument('--popularity_catalog_steps', type=int, default=0)
    parser.add_argument('--popularity_catalog_lr', type=float, default=0.001)
    parser.add_argument(
        '--popularity_catalog_anchor_coef', type=float, default=0.1
    )

    # customized args
    parser = modelClass.parse_model_args(parser)
    parser = readerClass.parse_data_args(parser)
    args, _ = parser.parse_known_args()
    print(args)

    utils.set_random_seed(args.seed)

    # 数据处理类
    reader = readerClass(args)
    print('data statistics:\n', reader.get_statistics())

    # cuda
    if args.cuda >= 0 and torch.cuda.is_available():
        visible_device_count = torch.cuda.device_count()
        if args.cuda >= visible_device_count:
            raise ValueError(
                f"CUDA device {args.cuda} is unavailable; "
                f"torch.cuda.device_count() is {visible_device_count}. "
                "If CUDA_VISIBLE_DEVICES is set, pass the logical device index."
            )
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    args.device = device

    # model and optimizer
    # 加载用户反馈模型和优化器
    model = modelClass(args, reader.get_statistics(), device)
    model = model.to(device)
    if args.init_checkpoint:
        print(f"Warm-start model weights from {args.init_checkpoint}")
        initial_checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu"
        )
        incompatible = model.load_state_dict(
            initial_checkpoint["model_state_dict"], strict=False
        )
        allowed_missing = (
            {"popularityDirection"}
            if args.popularity_loss_coef > 0.0 else set()
        )
        actual_missing = set(incompatible.missing_keys)
        if actual_missing not in (set(), allowed_missing):
            raise ValueError(
                "unexpected warm-start missing keys: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise ValueError(
                "unexpected warm-start keys: "
                f"{incompatible.unexpected_keys}"
            )
        del initial_checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer

    try:
        best_val_auc = -np.inf
        best_val_report = None
        best_epoch = None
        print(f"validation before training:")
        val_report = do_eval(model, reader, args, phase="val")
        print(f"Val result:")
        print(val_report)

        epo = 0
        stop_count = 0
        if args.epoch == 0:
            if args.popularity_catalog_steps <= 0:
                raise ValueError(
                    "epoch=0 requires popularity_catalog_steps > 0"
                )
            print("catalog-only warm-start calibration")
            model.train()
            catalog_report = calibrate_item_catalog(model, reader, args)
            print(f"Final catalog calibration report: {catalog_report}")
            val_report = do_eval(model, reader, args, phase="val")
            print("Val result after catalog calibration:")
            print(val_report)
            best_val_auc = val_report['mean_auc']
            best_val_report = val_report
            best_epoch = 0
            model.save_checkpoint()
        while epo < args.epoch:
            epo += 1
            print(f"epoch {epo} training")

            # train an epoch
            model.train()
            reader.set_phase("train")
            train_loader = DataLoader(reader, batch_size=args.batch_size,
                                      shuffle=True, pin_memory=True,
                                      num_workers=reader.n_worker)
            t1 = time()
            pbar = tqdm(total=len(reader))
            step_loss = []
            step_popularity_loss = []
            step_behavior_loss = {fb: [] for fb in model.feedback_types}
            for i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                # 把数据转换为tensor类型
                wrapped_batch = utils.wrap_batch(batch_data, device=device)
                if epo == 1 and i == 0:
                    utils.show_batch(wrapped_batch)
                out_dict = model.do_forward_and_loss(wrapped_batch)
                loss = out_dict['loss']
                loss.backward()
                step_loss.append(loss.item())
                step_popularity_loss.append(out_dict.get('popularity_loss', 0.0))
                for fb, v in out_dict['behavior_loss'].items():
                    step_behavior_loss[fb].append(v)
                optimizer.step()
                pbar.update(args.batch_size)
                if i % 100 == 0:
                    print(f"Iteration {i}, loss: {np.mean(step_loss[-100:])}")
                    if args.popularity_loss_coef > 0.0:
                        print(
                            "Popularity auxiliary loss: "
                            f"{np.mean(step_popularity_loss[-100:])}"
                        )
                    print({fb: np.mean(v[-100:]) for fb, v in step_behavior_loss.items()})
            pbar.close()
            catalog_report = calibrate_item_catalog(model, reader, args)
            if catalog_report is not None:
                print(f"Final catalog calibration report: {catalog_report}")
            print("Epoch {}; time {:.4f}".format(epo, time() - t1))

            # validation
            t2 = time()
            print(f"epoch {epo} validating")
            val_report = do_eval(model, reader, args, phase="val")
            print(f"Val result:")
            print(val_report)

            # Select one checkpoint using a single pre-declared aggregate metric.
            # The environment-specific test split remains untouched until training ends.
            if args.save_with_val:
                if val_report['mean_auc'] > best_val_auc:
                    best_val_auc = val_report['mean_auc']
                    best_val_report = val_report
                    best_epoch = epo
                    model.save_checkpoint()
                    stop_count = 0
                else:
                    stop_count += 1
                if stop_count >= args.early_stop_patience:
                    print(f"Early stopping after {stop_count} epochs without mean-AUC improvement")
                    break
            else:
                model.save_checkpoint()

    except KeyboardInterrupt:
        print("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            print(os.linesep + '-' * 20 + ' END: ' + utils.get_local_time() + ' ' + '-' * 20)
            exit(1)

    checkpoint_path = args.model_path + ".checkpoint"
    if os.path.isfile(checkpoint_path):
        # Training is finished, so Adam state is no longer needed.  Release it
        # before reloading the best checkpoint for the held-out evaluation.
        model.optimizer = None
        optimizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.load_from_checkpoint(args.model_path, with_optimizer=False)
        print("final held-out test evaluation:")
        test_report = do_eval(model, reader, args, phase="test")
        print("Test result:")
        print(test_report)
        metrics_path = args.model_path + ".metrics.json"
        metrics = {
            "schema_version": 1,
            "seed": args.seed,
            "environment_split": args.environment_split,
            "init_checkpoint": args.init_checkpoint or None,
            "epochs_completed": epo,
            "best_epoch": best_epoch,
            "best_validation_mean_auc": float(best_val_auc),
            "best_validation_auc": {
                key: float(value)
                for key, value in best_val_report["auc"].items()
            },
            "held_out_test_mean_auc": float(test_report["mean_auc"]),
            "held_out_test_auc": {
                key: float(value) for key, value in test_report["auc"].items()
            },
            "popularity_loss_coef": float(args.popularity_loss_coef),
            "popularity_margin": float(args.popularity_margin),
            "popularity_catalog_steps": args.popularity_catalog_steps,
            "popularity_catalog_lr": args.popularity_catalog_lr,
            "popularity_catalog_anchor_coef": (
                args.popularity_catalog_anchor_coef
            ),
            "item_metadata_alignment": reader.item_metadata_alignment,
        }
        with open(metrics_path, "w", encoding="utf-8") as outfile:
            json.dump(metrics, outfile, indent=2, sort_keys=True)
            outfile.write("\n")
        print(f"Machine-readable metrics saved to {metrics_path}")
    else:
        raise FileNotFoundError(f"No trained checkpoint was produced at {checkpoint_path}")
