import os
import argparse
import sys
import logging
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
from torch.optim import Adam

# from torch.utils.tensorboard import SummaryWriter
from hssm_v2 import EnvModel
from utils import (
    preprocess,
    postprocess,
    full_dataloader_toy,
    log_train,
    log_test,
    plot_rec,
    plot_gen,
)
from datetime import datetime
import wandb

LOGGER = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description="vta agr parser")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--name", type=str, default="colors")

    # data size
    parser.add_argument("--dataset-path", type=str, default="./data/toy_data.npy")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-size", type=int, default=18)
    parser.add_argument("--init-size", type=int, default=3)

    # model size
    parser.add_argument("--state-size", type=int, default=64)
    parser.add_argument("--belief-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)

    # observation distribution
    parser.add_argument("--obs-std", type=float, default=1.0)
    parser.add_argument("--obs-bit", type=int, default=5)

    # optimization
    parser.add_argument("--learn-rate", type=float, default=0.0005)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--max-iters", type=int, default=100000)

    # subsequence prior params
    parser.add_argument("--seg-num", type=int, default=5)
    parser.add_argument("--seg-len", type=int, default=8)

    # gumbel params
    parser.add_argument("--max-beta", type=float, default=1.0)
    parser.add_argument("--min-beta", type=float, default=0.1)
    parser.add_argument("--beta-anneal", type=float, default=100)

    # log dir
    parser.add_argument("--log-dir", type=str, default="./asset/log/")

    # coding length params
    parser.add_argument("--use_abs_pos_kl", type=float, default=0)
    parser.add_argument("--coding_len_coeff", type=float, default=1.0)
    return parser.parse_args()


def date_str():
    s = str(datetime.now())
    d, t = s.split(" ")
    t = "-".join(t.split(":")[:-1])
    return d + "-" + t


def set_exp_name(args):
    exp_name = args.name + '_' + date_str()
    return exp_name


def main():
    # parse arguments
    args = parse_args()

    print(">"*80)
    print(args)
    print(">"*80)

    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # set logger
    log_format = "[%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)

    # set size
    seq_size = args.seq_size
    init_size = args.init_size

    # set device as gpu
    device = torch.device("cuda", 0)

    # set writer
    exp_name = set_exp_name(args)

    wandb.init(
        project="mdl",
        entity="ydjiang",
        name=exp_name,
        sync_tensorboard=False,
        settings=wandb.Settings(start_method="fork"),
    )

    LOGGER.info("EXP NAME: " + exp_name)
    # load dataset
    train_loader, test_loader = full_dataloader_toy(
        seq_size, init_size, args.batch_size, path=args.dataset_path
    )
    LOGGER.info("Dataset loaded")

    # init models
    model = EnvModel(
        belief_size=args.belief_size,
        state_size=args.state_size,
        num_layers=args.num_layers,
        max_seg_len=args.seg_len,
        max_seg_num=args.seg_num,
        use_abs_pos_kl=args.use_abs_pos_kl == 1.0,
        coding_len_coeff=args.coding_len_coeff,
    ).to(device)
    LOGGER.info("Model initialized")

    # init optimizer
    optimizer = Adam(params=model.parameters(), lr=args.learn_rate, amsgrad=True)

    # test data
    pre_test_full_data_list = iter(test_loader).next()
    pre_test_full_data_list = preprocess(
        pre_test_full_data_list.to(device), args.obs_bit
    )

    # for each iter
    b_idx = 0
    while b_idx <= args.max_iters:
        # for each batch
        for train_obs_list in train_loader:
            b_idx += 1
            # mask temp annealing
            if args.beta_anneal:
                model.state_model.mask_beta = (
                    args.max_beta - args.min_beta
                ) * 0.999 ** (b_idx / args.beta_anneal) + args.min_beta
            else:
                model.state_model.mask_beta = args.max_beta

            ##############
            # train time #
            ##############
            # get input data
            train_obs_list = preprocess(train_obs_list.to(device), args.obs_bit)

            # run model with train mode
            model.train()
            optimizer.zero_grad()
            results = model(train_obs_list, seq_size, init_size, args.obs_std)

            # get train loss and backward update
            train_total_loss = results["train_loss"]
            train_total_loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Hack for computing F1
            boundaries = results['mask_data'].squeeze().detach().cpu().numpy()
            true_boundaries = np.zeros(boundaries.shape)
            true_boundaries[:, ::3] = 1.0
            precision = (boundaries * true_boundaries).sum() / boundaries.sum()
            recall = (boundaries * true_boundaries).sum() / true_boundaries.sum()
            f1 = 2./(1./precision + 1./recall)
            results['precision'] = precision
            results['recall'] = recall
            results['f1'] = f1

            # log
            if b_idx % 10 == 0:
                train_stats, log_str, log_data = log_train(results, None, b_idx)
                LOGGER.info(log_str, *log_data)
                wandb.log(train_stats, step=b_idx)

            #############
            # test time #
            #############
            if b_idx % 100 == 0:
                # set data
                pre_test_init_data_list = pre_test_full_data_list[:, :init_size]
                post_test_init_data_list = postprocess(
                    pre_test_init_data_list, args.obs_bit
                )
                pre_test_input_data_list = pre_test_full_data_list[
                    :, init_size : (init_size + seq_size)
                ]
                post_test_input_data_list = postprocess(
                    pre_test_input_data_list, args.obs_bit
                )

                with torch.no_grad():
                    ##################
                    # test data elbo #
                    ##################
                    model.eval()
                    results = model(
                        pre_test_full_data_list, seq_size, init_size, args.obs_std
                    )
                    post_test_rec_data_list = postprocess(
                        results["rec_data"], args.obs_bit
                    )
                    output_img, output_mask = plot_rec(
                        post_test_init_data_list,
                        post_test_input_data_list,
                        post_test_rec_data_list,
                        results["mask_data"],
                        results["p_mask"],
                        results["q_mask"],
                        results["option_list"],
                    )

                    # Hack for computing F1
                    boundaries = results['mask_data'].squeeze().detach().cpu().numpy()
                    true_boundaries = np.zeros(boundaries.shape)
                    true_boundaries[:, ::3] = 1.0
                    precision = (boundaries * true_boundaries).sum() / boundaries.sum()
                    recall = (boundaries * true_boundaries).sum() / true_boundaries.sum()
                    f1 = 2./(1./precision + 1./recall)
                    results['precision'] = precision
                    results['recall'] = recall
                    results['f1'] = f1

                    # log
                    test_stats, log_str, log_data = log_test(results, None, b_idx)
                    LOGGER.info(log_str, *log_data)
                    wandb.log(test_stats, step=b_idx)

                    wandb.log(
                        {
                            "valid_image/rec_image": wandb.Image(output_img),
                            "valid_image/mask_image": wandb.Image(output_mask),
                        },
                        step=b_idx,
                    )

                    ###################
                    # full generation #
                    ###################
                    (
                        pre_test_gen_data_list,
                        test_mask_data_list,
                        option_list,
                    ) = model.full_generation(
                        pre_test_init_data_list, seq_size
                    )  # TODO: return the option indices
                    post_test_gen_data_list = postprocess(
                        pre_test_gen_data_list, args.obs_bit
                    )

                    # log
                    output_img = plot_gen(
                        post_test_init_data_list,
                        post_test_gen_data_list,
                        test_mask_data_list,
                        option_list,
                    )  # TODO: add number to the generated

                    wandb.log(
                        {"valid_image/full_gen_image": wandb.Image(output_img),},
                        step=b_idx,
                    )


if __name__ == "__main__":
    main()
