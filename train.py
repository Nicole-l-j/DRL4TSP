import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, problem, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    gt_cost = rollout_groundtruth(problem, dataset, opts)
    opt_gap = ((cost/gt_cost - 1) * 100)
    print('Validation groundtruth cost: {:.3f} +- {:.3f}'.format(
        gt_cost.mean(), torch.std(gt_cost)))
    print('Validation overall avg_cost: {:.3f} +- {:.3f}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    print('Validation optimality gap: {:.3f}% +- {:.3f}'.format(
        opt_gap.mean(), torch.std(opt_gap)))

    return avg_cost, opt_gap.mean()


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat['nodes'], opts.device), move_to(bat['graph'], opts.device), opts)
        cost, _ = torch.min(cost, 1)
        # cost = cost[:,0]
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(
            DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers),
            disable=opts.no_progress_bar, ascii=True)
    ], 0)


def rollout_groundtruth(problem, dataset, opts):
    return torch.cat([
        problem.get_costs(bat['nodes'], bat['tour_nodes'])[0]
        for bat in DataLoader(
            dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_datasets, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch)

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(
            min_size=opts.min_size, max_size=opts.max_size, batch_size=opts.batch_size,
            num_samples=opts.epoch_size, distribution=opts.data_distribution,
            neighbors=opts.neighbors, knn_strat=opts.knn_strat
        ))
    training_dataloader = DataLoader(
        training_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    # Put model in train mode!
    model.train()
    optimizer.zero_grad()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    # avg_reward = validate(model, val_datasets, opts)
    for val_idx, val_dataset in enumerate(val_datasets):
        avg_reward, avg_opt_gap = validate(model, val_dataset, problem, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward', avg_reward, step)
            tb_logger.log_value('val_opt_gap', avg_opt_gap, step)

    baseline.epoch_callback(model, epoch)


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):

    optimizer.zero_grad()
    # Unwrap baseline
    bat, bl_val = baseline.unwrap_batch(batch)

    # Optionally move Tensors to GPU
    x = move_to(bat['nodes'], opts.device)
    graph = move_to(bat['graph'], opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    costs, log_likelihood, loss, = model(x, graph, opts, baseline, bl_val,return_pi=False)

    costs, _ = torch.min(costs, 1)

    # Calculate total loss of all decoders
    loss = loss.sum()
    # Perform backward pass
    loss.backward()

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    if grad_norms[0][0]!=grad_norms[0][0]:
        optimizer.zero_grad()
        print("nan detected")
        return

    # Perform optimization step after accumulating gradients
    if step % opts.accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(costs, grad_norms, epoch, batch_id, step,
                   log_likelihood, log_likelihood.mean(), 0, tb_logger, opts)
