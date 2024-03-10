import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timeit import default_timer
from attrdict import AttrDict

from task_configs import get_data, get_config, get_metric, get_optimizer_scheduler
from utils import count_params, count_trainable_params, denormalize
from embedder import get_tgt_model


def main(use_determined, args, info=None, context=None):
    print(args)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = '/datasets' if use_determined else './datasets'

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)

    if args.reproducibility:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    sample_shape, loss, args = get_config(root, args)

    if load_embedder(use_determined, args):
        args.embedder_epochs = 0

    model, embedder_stats = get_tgt_model(args, root, sample_shape, loss, eval_mode=args.continue_train)
        
    train_loader, val_loader, test_loader, n_train, n_val, n_test = get_data(root, args.dataset, args.batch_size, args.valid_split, args=args)
    metric, compare_metrics = get_metric(root, args.dataset)
    
    model, ep_start, id_best, train_score, train_losses, embedder_stats_saved = load_state(use_determined, args, context, model, None, None, n_train, freq=args.validation_freq, test=True, dirname="best_")
    embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    
    offset = 0 if ep_start == 0 else 1
    args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None if args.predictor_epochs == 0 or ep_start >= args.predictor_epochs else 'predictor', n_train=n_train)
    train_full = args.predictor_epochs == 0 or ep_start >= args.predictor_epochs
    if ep_start == 0:
        save_state(use_determined, args, context, model, optimizer, scheduler, 0, n_train, train_score, train_losses, embedder_stats, "best_")
    
    if args.device == 'cuda':
        model.cuda()
        try:
            loss.cuda()
        except:
            pass

    print("\n------- Experiment Summary --------")
    print("id:", args.experiment_id)
    print("dataset:", args.dataset, "\tbatch size:", args.batch_size, "\tlr:", args.optimizer.params.lr)
    print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)
    print("param count:", count_params(model), count_trainable_params(model))
    #print(model)
    
    model, ep_start, id_best, train_score, train_losses, embedder_statssaved = load_state(use_determined, args, context, model, optimizer, scheduler, n_train, freq=args.validation_freq, dirname="best_")
    embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    train_time = []

    print("\n------- Start Training --------" if ep_start == 0 else "\n------- Resume Training --------", ep_start)

    for ep in range(ep_start, args.epochs + args.predictor_epochs):

        if not train_full and ep >= args.predictor_epochs:
            args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None, n_train=n_train)
            train_full = True

        time_start = default_timer()

        train_loss = train_one_epoch(context, args, model, optimizer, scheduler, train_loader, loss, n_train, ep=ep)
        train_time_ep = default_timer() - time_start 

        if ep % args.validation_freq == 0 or ep == args.epochs + args.predictor_epochs - 1: 
                
            val_loss, val_score = evaluate(context, args, model, val_loader, loss, metric, n_val)

            train_losses.append(train_loss)
            train_score.append(val_score)
            train_time.append(train_time_ep)

            print("[train", "full" if ep >= args.predictor_epochs else "predictor", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))

            if use_determined:
                id_current = save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
                try:
                    context.train.report_training_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"train loss": train_loss, "epoch time": train_time_ep})
                    context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"val score": val_score})
                except:
                    pass
                    
            if compare_metrics(train_score) == val_score:
                if not use_determined:
                    id_current = save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats, "best_")
                id_best = id_current
 
    if ep == args.epochs + args.predictor_epochs - 1:
        eval_multiple(use_determined, context, root, args, model, loss, metric, optimizer, scheduler, n_train, id_best)

    if use_determined and context.preempt.should_preempt():
        print("paused")
        return
    

def train_one_epoch(context, args, model, optimizer, scheduler, loader, loss, temp, ep=0):    

    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(loader):

        x, y = data
        
        if isinstance(x, list):
            x, text_embeddings = x
            text_embeddings = text_embeddings.to(args.device)
        else:
            text_embeddings = None

        if isinstance(y, list):
            y, mask = y
            y = y.to(args.device)
            mask = mask.to(args.device)
            y *= mask
        else:
            y = y.to(args.device)
            mask = None

        x = x.to(args.device) 
        out = model(x, text_embeddings=text_embeddings)

        if mask is not None:
            out *= mask

        l = loss(out, y)
        l.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if (i + 1) % args.accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if args.lr_sched_iter:
            scheduler.step()

        train_loss += l.item()

        if i >= temp - 1:
            break

    if (not args.lr_sched_iter):
        scheduler.step()

    return train_loss / temp


def eval_multiple(use_determined, context, root, args, model, loss, metric, optimizer, scheduler, n_train, id_best):
    print("\n------- Start Test --------")

    for pde_subset in ["Burgers", "1DCFD", "ADV", "DS", "SW", "2DCFD","RD","RD2D", "NS"]:
        args.pde_subset = pde_subset

        _, _, test_loader, _, _, n_test = get_data(root, args.dataset, args.batch_size, args.valid_split, args=args)

        print(args.pde_subset)
        test_model = model
        test_time_start = default_timer()
        test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test)
        test_time_end = default_timer()

        print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            
        test_model, _, _, _, _, _ = load_state(use_determined, args, context, test_model, optimizer, scheduler, n_train, id_best, test=True, dirname="best_")
        test_time_start = default_timer()
        test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test)
        test_time_end = default_timer()

        print("[test best]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            


def evaluate(context, args, model, loader, loss, metric, n_eval):
    model.eval()
    
    eval_loss, eval_score = 0, 0
    
    ys, outs, n_eval, n_data = [], [], 0, 0
    masks, means, stds = [], [], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            x, y = data
            
            if isinstance(x, list):
                x, text_embeddings = x
                text_embeddings = text_embeddings.to(args.device)
            else:
                text_embeddings = None

            if isinstance(y, list):
                y, mask = y
                y = y.to(args.device)
                mask = mask.to(args.device)
                y *= mask
                masks.append(mask)
            else:
                y = y.to(args.device)
                mask = None

            x = x.to(args.device)
            out = model(x, text_embeddings=text_embeddings)

            if mask is not None:
                out *= mask

            ys.append(y)
            outs.append(out)
            n_data += x.shape[0]

            if n_data >= args.eval_batch_size or i == len(loader) - 1:
                outs = torch.cat(outs, 0)
                ys = torch.cat(ys, 0)
                masks = torch.cat(masks, 0)

                outs, ys = denormalize(outs, ys, loader.dataset.mean, loader.dataset.std)

                outs *= masks
                ys *= masks

                eval_loss += loss(outs, ys).item()
                eval_score += metric(outs, ys).item()
                n_eval += 1

                ys, outs, n_data = [], [], 0
                masks = []

        eval_loss /= n_eval
        eval_score /= n_eval

    return eval_loss, eval_score


########################## Helper Funcs ##########################

def save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats, dirname=""):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.experiment_id) + "/" + dirname + str(args.seed)
        if not os.path.exists(path):
            os.makedirs(path)
        
        save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
        return ep

    else:
        checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
        with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
            save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
            return uuid


def save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats):
    np.save(os.path.join(path, 'hparams.npy'), args)
    np.save(os.path.join(path, 'train_score.npy'), train_score)
    np.save(os.path.join(path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(path, 'embedder_stats.npy'), embedder_stats)

    model_state_dict = {
                'network_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
    torch.save(model_state_dict, os.path.join(path, 'state_dict.pt'))

    rng_state_dict = {
                'cpu_rng_state': torch.get_rng_state(),
                'gpu_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'py_rng_state': random.getstate()
            }
    torch.save(rng_state_dict, os.path.join(path, 'rng_state.ckpt'))


def load_embedder(use_determined, args):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.experiment_id) + "/best_" + str(args.seed)
        return os.path.isfile(os.path.join(path, 'state_dict.pt'))
    else:

        info = det.get_cluster_info()
        checkpoint_id = info.latest_checkpoint
        return checkpoint_id is not None


def load_state(use_determined, args, context, model, optimizer, scheduler, n_train, checkpoint_id=None, test=False, freq=1, dirname=""):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.experiment_id) + "/" + dirname + str(args.seed)
        if not os.path.isfile(os.path.join(path, 'state_dict.pt')):
            return model, 0, 0, [], [], None
    else:

        if checkpoint_id is None:
            info = det.get_cluster_info()
            checkpoint_id = info.latest_checkpoint
            if checkpoint_id is None:
                return model, 0, 0, [], [], None
        
        checkpoint = client.get_checkpoint(checkpoint_id)
        path = checkpoint.download()

    train_score = np.load(os.path.join(path, 'train_score.npy'))
    train_losses = np.load(os.path.join(path, 'train_losses.npy'))
    embedder_stats = np.load(os.path.join(path, 'embedder_stats.npy'))
    epochs = freq * (len(train_score) - 1) + 1
    checkpoint_id = checkpoint_id if use_determined else epochs - 1
    model_state_dict = torch.load(os.path.join(path, 'state_dict.pt'))
    model.load_state_dict(model_state_dict['network_state_dict'])
    
    if not test:
        optimizer.load_state_dict(model_state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(model_state_dict['scheduler_state_dict'])

        rng_state_dict = torch.load(os.path.join(path, 'rng_state.ckpt'), map_location='cpu')
        torch.set_rng_state(rng_state_dict['cpu_rng_state'])
        torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
        np.random.set_state(rng_state_dict['numpy_rng_state'])
        random.setstate(rng_state_dict['py_rng_state'])

        if use_determined: 
            try:
                for ep in range(epochs):
                    if ep % freq == 0:
                        context.train.report_training_metrics(steps_completed=(ep + 1) * n_train, metrics={"train loss": train_losses[ep // freq]})
                        context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train, metrics={"val score": train_score[ep // freq]})
            except:
                print("load error")

    return model, epochs, checkpoint_id, list(train_score), list(train_losses), embedder_stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UPS')
    parser.add_argument('--config', type=str, default=None, help='config file name')
    parser.add_argument('--continue_train',type=bool, default=False)


    args = parser.parse_args()
    if args.config is not None:     
        import yaml

        with open(args.config, 'r') as stream:
            args_ = AttrDict(yaml.safe_load(stream)['hyperparameters'])
            args_.continue_train = args.continue_train
            main(False, args_)

    else:
        import determined as det
        from determined.experimental import client
        from determined.pytorch import DataLoader

        info = det.get_cluster_info()
        args = AttrDict(info.trial.hparams)
        
        with det.core.init() as context:
            main(True, args, info, context)