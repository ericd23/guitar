import os, time
import importlib
from collections import namedtuple

import env
from models import ACModel, Discriminator, AdaptNet
from utils import seed

import torch
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str,
    help="Configure file used for training. Please refer to files in `config` folder.")
parser.add_argument("--ckpt", type=str, default=None,
    help="Checkpoint directory or file for training or evaluation.")
parser.add_argument("--test", action="store_true", default=False,
    help="Run visual evaluation.")
parser.add_argument("--seed", type=int, default=42,
    help="Random seed.")
parser.add_argument("--device", type=int, default=0,
    help="ID of the target GPU device for model running.")
parser.add_argument("--silent", action="store_true", default=False,
    help="Silent model during training.")

parser.add_argument("--note", type=str, default=None,
    help="Specific musci note policy training or evaluation.")

parser.add_argument("--left", type=str, default=None,
    help="Checkpoint directory or file for left-hand policy training or evaluation.")
parser.add_argument("--right", type=str, default=None,
    help="Checkpoint directory or file for right-hand policy training or evaluation.")

parser.add_argument("--headless", action="store_true", default=False,
                    help="Run in headless mode with no viewer window.")
parser.add_argument("--record", type=str, default=None,
                    help="Output .mp4 file to record the environment in headless mode.")
parser.add_argument("--width", type=int, default=1920, # Default width
                    help="Output video width in pixels for headless recording.")
parser.add_argument("--height", type=int, default=1080, # Default height
                    help="Output video height in pixels for headless recording.")
parser.add_argument("--frames", type=int, default=None, # No default frame limit
                    help="Total number of frames to simulate and record in headless mode.")
parser.add_argument("--fps", type=int, default=60, # Default FPS
                    help="Recording frame rate (frames per second) for headless recording. Also affects simulation dt if recording.")

settings = parser.parse_args()

TRAINING_PARAMS = dict(
    horizon = 8,
    num_envs = 512,
    batch_size = 256,
    opt_epochs = 5,
    actor_lr = 5e-6,
    critic_lr = 1e-4,
    gamma = 0.95,
    lambda_ = 0.95,
    disc_lr = 1e-5,
    max_epochs = 50000,
    save_interval = 10000,
    log_interval = 50,
    terminate_reward = -25,
    control_mode = "position",
)

def test(env, model, total_frames=None): # Add total_frames parameter
    model.eval()
    env.eval()
    env.reset()
    # Initialize lists to store metrics *since last print*
    accuracy_l, precision_l, recall_l = [], [], []
    accuracy_r, precision_r, recall_r = [], [], []
    metrics_ready_to_print = False # Flag to print on next iteration after collecting
    nn = 0 # Note/Reset counter
    frame_count = 0 # <<< Initialize frame counter

    print("Starting evaluation loop...")
    if total_frames:
        print(f"Will stop after {total_frames} frames.")

    try: # Use try/finally to ensure env.close() is called
        while not env.request_quit: # Use env.request_quit for potential external close signals
            # --- (Metric printing logic remains the same) ---
            if metrics_ready_to_print:
                 metrics_printed_this_cycle = False
                 if precision_l: # Check if list has data
                     print(f"{nn} L Acc: {np.mean(accuracy_l):.4f}, Prec: {np.mean(precision_l):.4f}, Rec: {np.mean(recall_l):.4f}")
                     accuracy_l.clear(); precision_l.clear(); recall_l.clear() # Clear after printing
                     metrics_printed_this_cycle = True
                 if precision_r: # Check if list has data
                     print(f"{nn} R Acc: {np.mean(accuracy_r):.4f}, Prec: {np.mean(precision_r):.4f}, Rec: {np.mean(recall_r):.4f}")
                     accuracy_r.clear(); precision_r.clear(); recall_r.clear() # Clear after printing
                     metrics_printed_this_cycle = True
                 if metrics_printed_this_cycle:
                      metrics_ready_to_print = False # Reset flag


            obs, info = env.reset_done()
            dones_tensor = env.done # Get done status *after* reset_done

            # Increment note counter if a reset occurred (assuming 1 env in test)
            if dones_tensor.numel() > 0 and dones_tensor[0]:
                 nn += 1
                 # print(f"Environment reset detected. Note counter: {nn}") # Optional debug


            seq_len = info["ob_seq_lens"]
            # Get deterministic actions in test mode
            with torch.no_grad():
                 actions = model.act(obs, seq_len-1, stochastic=False)

            # <<< Step the environment >>>
            # Note: env.step() in HeadlessEnv now also handles frame writing
            obs_, rews, dones, info = env.step(actions) # dones here is the *new* done status

            # --- (Metric Collection logic remains the same) ---
            metrics_collected_this_step = False
            if "accuracy_l" in info and info["accuracy_l"].numel() > 0:
                accuracy_l.extend(info["accuracy_l"].cpu().tolist())
                metrics_collected_this_step = True
            if "precision_l" in info and info["precision_l"].numel() > 0:
                precision_l.extend(info["precision_l"].cpu().tolist())
                metrics_collected_this_step = True
            if "recall_l" in info and info["recall_l"].numel() > 0:
                recall_l.extend(info["recall_l"].cpu().tolist())
                metrics_collected_this_step = True

            if "accuracy_r" in info and info["accuracy_r"].numel() > 0:
                accuracy_r.extend(info["accuracy_r"].cpu().tolist())
                metrics_collected_this_step = True
            if "precision_r" in info and info["precision_r"].numel() > 0:
                precision_r.extend(info["precision_r"].cpu().tolist())
                metrics_collected_this_step = True
            if "recall_r" in info and info["recall_r"].numel() > 0:
                recall_r.extend(info["recall_r"].cpu().tolist())
                metrics_collected_this_step = True

            # Set the flag to print metrics on the *next* iteration if collected
            if metrics_collected_this_step:
                metrics_ready_to_print = True

            # <<< Increment frame counter >>>
            frame_count += 1

            # <<< Loop Termination Condition >>>
            if total_frames is not None and frame_count >= total_frames:
                print(f"\nReached target frame count: {frame_count}/{total_frames}. Stopping.")
                break # Exit the while loop

            # Optional: Check for viewer close request even in headless (might be useful?)
            # Handled by env.request_quit based on super().step() logic

    finally: # Ensure cleanup happens
        print("Evaluation loop finished or interrupted.")
        # env.close() will handle releasing video writer etc. via HeadlessEnv.close
        # Check if env exists and has close method before calling
        if 'env' in locals() and hasattr(env, 'close') and callable(getattr(env, 'close')):
             print("Closing environment and releasing resources...")
             env.close()
        else:
             print("Environment does not exist or does not have a close method.")
# <<< END MODIFIED test() FUNCTION >>>


def train(env, model, ckpt_dir, training_params):
    if ckpt_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(ckpt_dir)
    else:
        logger = None

    optimizer = torch.optim.Adam([
        {"params": model.actor.parameters(), "lr": training_params.actor_lr},
        {"params": model.critic.parameters(), "lr": training_params.critic_lr}
    ])

    ac_parameters = list(model.actor.parameters()) + list(model.critic.parameters())
    if model.discriminators:
        disc_optimizer = torch.optim.Adam(
            sum([list(disc.parameters()) for disc in model.discriminators.values()], []),
            training_params.disc_lr)
    epoch = 0

    buffer = dict(
        s=[], a=[], v=[], lp=[], v_=[], not_done=[], terminate=[],
        ob_seq_len=[]
    )
    multi_critics = env.reward_weights is not None and env.reward_weights.size(-1) > 1
    reward_weights = env.reward_weights if multi_critics else None
    has_goal_reward = env.rew_dim > 0
    if has_goal_reward: buffer["r"] = []

    buffer_disc = {
        name: dict(
            fake=[], real=[], #seq_len=[]
        ) for name in env.discriminators.keys()
    }
    # real_losses, fake_losses = {n:[] for n in buffer_disc.keys()}, {n:[] for n in buffer_disc.keys()}

    BATCH_SIZE = training_params.batch_size
    HORIZON = training_params.horizon
    GAMMA = training_params.gamma
    GAMMA_LAMBDA = training_params.gamma * training_params.lambda_
    OPT_EPOCHS = training_params.opt_epochs
    LOG_INTERVAL = training_params.log_interval

    model.eval()
    env.train()
    env.reset()
    tic = time.time()

    accuracy_l, precision_l, recall_l, f1_l = [], [], [], []
    accuracy_r, precision_r, recall_r, f1_r = [], [], [], []
    while not env.request_quit:
        with torch.no_grad():
            obs, info = env.reset_done()
            seq_len = info["ob_seq_lens"]
            actions, values, log_probs = model.act(obs, seq_len-1, stochastic=True)
            obs_, rews, dones, info = env.step(actions)
            log_probs = log_probs.sum(-1, keepdim=True)
            not_done = (~dones).unsqueeze_(-1)
            terminate = info["terminate"]

            if env.discriminators:
                fakes = info["disc_obs"]
                reals = info["disc_obs_expert"]

            values_ = model.evaluate(obs_, seq_len)

            if "accuracy_l" in info and info["accuracy_l"].numel() > 0:
                accuracy_l.append(info["accuracy_l"])
                precision_l.append(info["precision_l"])
                recall_l.append(info["recall_l"])
                # f1_l.extend(info["f1_l"])
            if "accuracy_r" in info and info["accuracy_r"].numel() > 0:
                accuracy_r.append(info["accuracy_r"])
                precision_r.append(info["precision_r"])
                recall_r.append(info["recall_r"])
                # f1_r.extend(info["f1_r"])

        buffer["s"].append(obs)
        buffer["a"].append(actions)
        buffer["v"].append(values)
        buffer["lp"].append(log_probs)
        buffer["v_"].append(values_)
        buffer["not_done"].append(not_done)
        buffer["terminate"].append(terminate)
        buffer["ob_seq_len"].append(seq_len)
        if has_goal_reward:
            buffer["r"].append(rews)
        if env.discriminators:
            for name, fake in fakes.items():
                buffer_disc[name]["fake"].append(fake)
                buffer_disc[name]["real"].append(reals[name])

        if len(buffer["s"]) == HORIZON:
            disc_data_training = []
            ob_seq_lens = torch.cat(buffer["ob_seq_len"])
            ob_seq_end_frames = ob_seq_lens - 1
            if env.discriminators:
                with torch.no_grad():
                    for name, data in buffer_disc.items():
                        disc = model.discriminators[name]
                        fake = torch.cat(data["fake"])
                        real_ = torch.cat(data["real"])
                        end_frame = ob_seq_lens # N

                        length = torch.arange(fake.size(1),
                            dtype=end_frame.dtype, device=end_frame.device
                        ).unsqueeze_(0)         # 1 x L
                        mask = length <= end_frame.unsqueeze(1)     # N x L

                        # let real samples (from ref motions) have the same length with the fake samples (simulated)
                        mask_ = length >= fake.size(1)-1 - end_frame.unsqueeze(1)
                        real = torch.zeros_like(real_)
                        real[mask] = real_[mask_]

                        disc.ob_normalizer.update(fake[mask])
                        disc.ob_normalizer.update(real[mask])
                        ob = disc.ob_normalizer(fake)
                        ref = disc.ob_normalizer(real)

                        disc_data_training.append((name, disc, ref, ob, end_frame))

                model.train()
                n_samples = len(fake)
                idx = torch.randperm(n_samples)
                # We assume all discriminator samples are drawn every timestep
                for batch in range(n_samples//BATCH_SIZE):
                    sample = idx[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                    for name, disc, ref, ob, seq_end_frame_ in disc_data_training:
                        # real_loss = real_losses[name]
                        # fake_loss = fake_losses[name]
                        r = ref[sample]
                        f = ob[sample]
                        seq_end_frame = seq_end_frame_[sample]
                        score_r = disc(r, seq_end_frame, normalize=False)
                        score_f = disc(f, seq_end_frame, normalize=False)
                        loss_r = torch.nn.functional.relu(1-score_r).mean()
                        loss_f = torch.nn.functional.relu(1+score_f).mean()

                        with torch.no_grad():
                            alpha = torch.rand(r.size(0), dtype=r.dtype, device=r.device)
                            alpha = alpha.view(-1, *([1]*(r.ndim-1)))
                            interp = alpha*r+(1-alpha)*f
                        interp.requires_grad = True
                        with torch.backends.cudnn.flags(enabled=False):
                            score_interp = disc(interp, seq_end_frame, normalize=False)
                        grad = torch.autograd.grad(
                            score_interp, interp, torch.ones_like(score_interp),
                            retain_graph=True, create_graph=True, only_inputs=True
                        )[0]
                        gp = grad.reshape(grad.size(0), -1).norm(2, dim=1).sub(1).square().mean()
                        l = loss_f + loss_r + 10*gp
                        l.backward()
                        # real_loss.append(score_r.mean().item())
                        # fake_loss.append(score_f.mean().item())
                    disc_optimizer.step()
                    disc_optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                terminate = torch.cat(buffer["terminate"])
                values = torch.cat(buffer["v"])
                values_ = torch.cat(buffer["v_"])
                log_probs = torch.cat(buffer["lp"])
                actions = torch.cat(buffer["a"])
                states = torch.cat(buffer["s"])

                if multi_critics:
                    rewards = torch.empty_like(values)
                else:
                    rewards = None
                for name, disc, _, ob, seq_end_frame in disc_data_training:
                    r = (disc(ob, seq_end_frame, normalize=False).clamp_(-1, 1)
                            .mean(-1, keepdim=True))
                    if rewards is None:
                        rewards = r
                    else:
                        rewards[:, env.discriminators[name].id] = r.squeeze_(-1)
                if has_goal_reward:
                    rewards_task = torch.cat(buffer["r"])
                    if rewards is None:
                        rewards = rewards_task
                    else:
                        rewards[:, -rewards_task.size(-1):] = rewards_task
                else:
                    rewards_task = None
                rewards[terminate] = training_params.terminate_reward

                if model.value_normalizer is not None:
                    values = model.value_normalizer(values, unnorm=True)
                    values_ = model.value_normalizer(values_, unnorm=True)
                values_[terminate] = 0
                rewards = rewards.view(HORIZON, -1, rewards.size(-1))
                values = values.view(HORIZON, -1, values.size(-1))
                values_ = values_.view(HORIZON, -1, values_.size(-1))

                not_done = buffer["not_done"]
                advantages = (rewards - values).add_(values_, alpha=GAMMA)
                for t in reversed(range(HORIZON-1)):
                    advantages[t].add_(advantages[t+1]*not_done[t], alpha=GAMMA_LAMBDA)

                advantages = advantages.view(-1, advantages.size(-1))
                returns = advantages + values.view(-1, advantages.size(-1))

                sigma, mu = torch.std_mean(advantages, dim=0, unbiased=True)
                advantages = (advantages - mu) / (sigma + 1e-8) # (HORIZON x N_ENVS) x N_DISC

                length = torch.arange(env.ob_horizon,
                    dtype=ob_seq_lens.dtype, device=ob_seq_lens.device)
                mask = length.unsqueeze_(0) < ob_seq_lens.unsqueeze(1)
                states_raw = model.observe(states, norm=False)[0]
                for normalizer in model.ob_normalizer_list:
                    normalizer.update(states_raw[mask])
                if model.value_normalizer is not None:
                    model.value_normalizer.update(returns)
                    returns = model.value_normalizer(returns)
                if multi_critics:
                    advantages.mul_(reward_weights)

            n_samples = advantages.size(0)
            epoch += 1
            model.train()
            # policy_loss, value_loss = [], []
            for _ in range(OPT_EPOCHS):
                idx = torch.randperm(n_samples)
                for batch in range(n_samples // BATCH_SIZE):
                    sample = idx[BATCH_SIZE * batch: BATCH_SIZE *(batch+1)]
                    s = states[sample]
                    a = actions[sample]
                    lp = log_probs[sample]
                    adv = advantages[sample]
                    v_t = returns[sample]
                    end_frame = ob_seq_end_frames[sample]

                    pi_, v_ = model(s, end_frame)
                    lp_ = pi_.log_prob(a).sum(-1, keepdim=True)
                    ratio = torch.exp(lp_ - lp)
                    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                    pg_loss = -torch.min(adv*ratio, adv*clipped_ratio).sum(-1).mean()

                    vf_loss = (v_ - v_t).square().mean()

                    loss = pg_loss + 0.5*vf_loss
                    loss.backward()
                    x = torch.nn.utils.clip_grad_norm_(ac_parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    # policy_loss.append(pg_loss.item())
                    # value_loss.append(vf_loss.item())
            model.eval()
            for v in buffer.values(): v.clear()
            for buf in buffer_disc.values():
                for v in buf.values(): v.clear()

            if epoch % LOG_INTERVAL == 1:
                acc_l = torch.mean(torch.cat(accuracy_l)) if accuracy_l else np.nan
                prec_l = torch.mean(torch.cat(precision_l)) if precision_l else np.nan
                rec_l = torch.mean(torch.cat(recall_l)) if recall_l else np.nan
                # f1_l = torch.mean(torch.cat(f1_l)) if f1_l else np.nan
                acc_r = torch.mean(torch.cat(accuracy_r)) if accuracy_r else np.nan
                prec_r = torch.mean(torch.cat(precision_r)) if precision_r else np.nan
                rec_r = torch.mean(torch.cat(recall_r)) if recall_r else np.nan
                # f1_r = torch.mean(torch.cat(f1_r)) if f1_r else np.nan
                if not settings.silent:
                    print("Epoch: {}, Acc: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f},  -- {:.4f}s".format(
                        epoch, acc_l, prec_l, rec_l, acc_r, prec_r, rec_r, time.time()-tic
                    ))
                if logger is not None:
                    # logger.add_scalar("train/lifetime", env.lifetime.to(torch.float32).mean().item(), epoch)
                    # r = rewards.view(-1, rewards.size(-1)).mean(0).cpu().tolist()
                    # logger.add_scalar("train/reward", np.mean(r), epoch)
                    # if rewards_task is not None:
                    #     rewards_task = rewards_task.mean(0).cpu().tolist()
                    #     for i in range(len(rewards_task)):
                    #         logger.add_scalar("train/task_reward_{}".format(i), rewards_task[i], epoch)
                    # logger.add_scalar("train/loss_policy", np.mean(policy_loss), epoch)
                    # logger.add_scalar("train/loss_value", np.mean(value_loss), epoch)
                    # for name, r_loss in real_losses.items():
                    #     if r_loss: logger.add_scalar("score_real/{}".format(name), sum(r_loss)/len(r_loss), epoch)
                    # for name, f_loss in fake_losses.items():
                    #     if f_loss: logger.add_scalar("score_fake/{}".format(name), sum(f_loss)/len(f_loss), epoch)
                    if accuracy_l:
                        logger.add_scalar("train/l_accuracy", acc_l, epoch)
                        logger.add_scalar("train/l_precision", prec_l, epoch)
                        logger.add_scalar("train/l_recall", rec_l, epoch)
                        # logger.add_scalar("train/l_f1", f1_l, epoch)
                    if accuracy_r:
                        logger.add_scalar("train/r_accuracy", acc_r, epoch)
                        logger.add_scalar("train/r_precision", prec_r, epoch)
                        logger.add_scalar("train/r_recall", rec_r, epoch)
                        # logger.add_scalar("train/r_f1", f1_r, epoch)
                accuracy_l.clear()
                precision_l.clear()
                recall_l.clear()
                # f1_l.clear()
                accuracy_r.clear()
                precision_r.clear()
                recall_r.clear()
                # f1_l.clear()
            # for v in real_losses.values(): v.clear()
            # for v in fake_losses.values(): v.clear()

            if ckpt_dir is not None:
                state = None
                if epoch % 500 == 0:
                    state = dict(model=model.state_dict())
                    torch.save(state, os.path.join(ckpt_dir, "ckpt"))
                if epoch % training_params.save_interval == 0:
                    if state is None:
                        state = dict(model=model.state_dict())
                    torch.save(state, os.path.join(ckpt_dir, "ckpt-{}".format(epoch)))
                if epoch >= training_params.max_epochs: exit()
            tic = time.time()

if __name__ == "__main__":
    if os.path.splitext(settings.config)[-1] in [".pkl", ".json", ".yaml"]:
        config = object()
        config.env_params = dict(
            motion_file = settings.config
        )
    else:
        spec = importlib.util.spec_from_file_location("config", settings.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

    seed(config.seed if hasattr(config, "seed") else settings.seed)

    if hasattr(config, "training_params"):
        TRAINING_PARAMS.update(config.training_params)
    if not TRAINING_PARAMS["save_interval"]:
        TRAINING_PARAMS["save_interval"] = TRAINING_PARAMS["max_epochs"]
    print(TRAINING_PARAMS)
    training_params = namedtuple('x', TRAINING_PARAMS.keys())(*TRAINING_PARAMS.values())

    # <<< START MODIFIED ENVIRONMENT INSTANTIATION (main block) >>>
    env_params_from_config = config.env_params if hasattr(config, "env_params") else {}
    env_kwargs = env_params_from_config.copy() # Start with config params

    # Check for discriminators in config
    discriminators_cfg = {}
    if hasattr(config, "discriminators"):
        # Assuming env.DiscriminatorConfig is defined elsewhere correctly
        discriminators_cfg = {
            name: env.DiscriminatorConfig(**prop)
            for name, prop in config.discriminators.items()
        }

    # Determine env_cls based on config (as before)
    from env import ICCGANHumanoid
    if hasattr(config, "env_cls"):
        # Make sure 'env' module is imported where needed if this is the base script
        # Assuming env = importlib.import_module("env") or similar is done
        env_cls_name = config.env_cls
        try:
            # Attempt to get the class from the 'env' module or globally
            if hasattr(env, env_cls_name):
                env_cls = getattr(env, env_cls_name)
            else:
                # Fallback to checking global scope (if env classes are defined in the main script)
                env_cls = globals()[env_cls_name]
                # Check if the found object is actually a class
                import inspect
                if not inspect.isclass(env_cls):
                    raise AttributeError
            print(f"Using environment class from config: {env_cls_name}")
        except (AttributeError, KeyError):
            print(f"Warning: Could not find environment class '{env_cls_name}' specified in config. Falling back to default.")
            # <<< IMPORTANT: Default should now check headless flag >>>
            # Assuming ICCGANHumanoid is the intended class family
            # Since ICCGANHumanoid now inherits from HeadlessEnv, we use it directly
            # but the constructor will handle the headless specifics.
            env_cls = ICCGANHumanoid # Default Class
    else:
        # <<< Default also needs to consider headless >>>
        env_cls = ICCGANHumanoid # Default Class
        print(f"Using default environment class: {env_cls.__name__}")

    # --- Add headless/recording specific params to kwargs ---
    env_kwargs["fps"] = settings.fps # Pass FPS to env constructor

    if settings.headless:
        env_kwargs["record"] = settings.record # Pass record path only if headless
        if not settings.record:
            print("Warning: --headless specified without --record. No video will be saved.")
        env_kwargs["cam_width"] = settings.width
        env_kwargs["cam_height"] = settings.height
        # HeadlessEnv constructor forces graphics_device = compute_device
        graphics_device_id = settings.device # Use compute device for graphics in headless
        print("Running in HEADLESS mode.")
    else:
        # For GUI mode, graphics device can be different, or None for default behavior
        graphics_device_id = None # Let IsaacGym manage default GUI window placement or use compute device
        print("Running in GUI mode.")
        if settings.record:
            print("Warning: --record specified without --headless. Recording is only supported in headless mode. Ignoring --record.")
        # Don't pass record/width/height if not headless

    # Handle --note argument (remains the same)
    if settings.note is not None:
        env_kwargs["note_file"] = settings.note
        if settings.test:
            env_kwargs["random_note_sampling"] = False

    # Set number of environments and episode length based on test mode
    if settings.test:
        num_envs = 1
        # Use --frames for episode length ONLY if recording, otherwise use a large default
        if settings.headless and settings.record and settings.frames:
            env_kwargs["episode_length"] = settings.frames
            print(f"Setting episode length based on --frames: {settings.frames}")
        else:
            env_kwargs["episode_length"] = 500000 # Large default if not recording or no frame limit
    else: # Training mode
        num_envs = training_params.num_envs
        # Keep the check for existing checkpoint during training
        if settings.ckpt and not settings.test and \
           (os.path.isfile(settings.ckpt) or \
            (os.path.isdir(settings.ckpt) and os.path.exists(os.path.join(settings.ckpt, "ckpt")))):
            raise ValueError(f"Checkpoint folder {settings.ckpt} exists. Add `--test` option to run test or remove checkpoint.")

    print(f"Instantiating environment: {env_cls.__name__} with {num_envs} environments.")
    # print(f"Environment Kwargs being passed: {env_kwargs}") # Optional: for debugging

    # --- Instantiate the environment ---
    # The selected env_cls (e.g., ICCGANHumanoid inheriting from HeadlessEnv)
    # will handle the specifics based on the passed kwargs.
    env = env_cls(n_envs=num_envs,
                  discriminators=discriminators_cfg,
                  compute_device=settings.device,
                  graphics_device=graphics_device_id, # Pass determined graphics device
                  **env_kwargs # Pass all collected kwargs (includes fps, record, width, height if headless)
                 )

    value_dim = len(env.discriminators)+env.rew_dim
    model = ACModel(env.state_dim, env.act_dim, env.goal_dim, value_dim)
    discriminators = torch.nn.ModuleDict({
        name: Discriminator(dim) for name, dim in env.disc_dim.items()
    })
    if settings.test:
        env.episode_length = 500000

    if "Two" in env_cls.__name__:
        assert (settings.left and settings.right) or settings.ckpt

        if settings.ckpt:
            if os.path.isdir(settings.ckpt):
                two = os.path.join(settings.ckpt, "ckpt")
            else:
                two = settings.ckpt
            if os.path.exists(two):
                state_dict_two = torch.load(two, map_location=torch.device(settings.device))

        if settings.left:
            if os.path.isdir(settings.left):
                ckpt = os.path.join(settings.left, "ckpt")
            else:
                ckpt = settings.left
            print("Load model from {}".format(ckpt))
            state_dict = torch.load(ckpt, map_location=torch.device(settings.device))
            state_dim = state_dict["model"]["ob_normalizer.mean"].shape[0]
            act_dim = state_dict["model"]["actor.mu.bias"].shape[0]
            goal_dim = state_dict["model"]["actor.embed_goal.0.weight"].shape[1]
            latent_dim = state_dict["model"]["actor.mlp.0.weight"].shape[1]
            left_policy = ACModel.Actor(state_dim, act_dim, goal_dim, latent_dim)
            policy_state, ob_state = dict(), dict()
            for k, v in state_dict["model"].items():
                if k.startswith("actor."):
                    policy_state[k[6:]] = v
                if k.startswith("ob_normalizer."):
                    ob_state[k[14:]] = v
            left_policy.load_state_dict(policy_state)
            left_policy.ob_normalizer = model.ob_normalizer.__class__(state_dim, clamp=5.0)
            left_policy.ob_normalizer.load_state_dict(ob_state)
        else:
            state_dim = state_dict_two["model"]["actor.meta1.ob_normalizer.mean"].shape[0]
            act_dim = state_dict_two["model"]["actor.meta1.mu.bias"].shape[0]
            goal_dim = state_dict_two["model"]["actor.meta1.embed_goal.0.weight"].shape[1]
            latent_dim = state_dict_two["model"]["actor.meta1.mlp.0.weight"].shape[1]
            left_policy = ACModel.Actor(state_dim, act_dim, goal_dim, latent_dim)
            left_policy.ob_normalizer = model.ob_normalizer.__class__(state_dim, clamp=5.0)

        if settings.right:
            if os.path.isdir(settings.right):
                ckpt = os.path.join(settings.right, "ckpt")
            else:
                ckpt = settings.right
            print("Load model from {}".format(ckpt))
            state_dict = torch.load(ckpt, map_location=torch.device(settings.device))
            state_dim = state_dict["model"]["ob_normalizer.mean"].shape[0]
            act_dim = state_dict["model"]["actor.mu.bias"].shape[0]
            goal_dim = state_dict["model"]["actor.embed_goal.0.weight"].shape[1]
            latent_dim = state_dict["model"]["actor.mlp.0.weight"].shape[1]
            policy_state, ob_state = dict(), dict()
            for k, v in state_dict["model"].items():
                if k.startswith("actor."):
                    policy_state[k[6:]] = v
                if k.startswith("ob_normalizer."):
                    ob_state[k[14:]] = v
            right_policy = ACModel.Actor(state_dim, act_dim, goal_dim, latent_dim)
            right_policy.load_state_dict(policy_state)
            right_policy.ob_normalizer = model.ob_normalizer.__class__(state_dim, clamp=5.0)
            right_policy.ob_normalizer.load_state_dict(ob_state)
        else:
            state_dim = state_dict_two["model"]["actor.meta2.ob_normalizer.mean"].shape[0]
            act_dim = state_dict_two["model"]["actor.meta2.mu.bias"].shape[0]
            goal_dim = state_dict_two["model"]["actor.meta2.embed_goal.0.weight"].shape[1]
            latent_dim = state_dict_two["model"]["actor.meta2.mlp.0.weight"].shape[1]
            right_policy = ACModel.Actor(state_dim, act_dim, goal_dim, latent_dim)
            right_policy.ob_normalizer = model.ob_normalizer.__class__(state_dim, clamp=5.0)

        # we let the reward weights consistent with the single-hand policy training,
        # i.e. the sum of weights are 2 instead of 1
        # this modification is trivial because we will perform grad clipping during training.
        env.reward_weights *= 2
        model.actor = AdaptNet(model, left_policy, right_policy)
    elif "Left" in env_cls.__name__ and settings.left and not settings.ckpt:
        settings.ckpt = settings.left
    elif "Right" in env_cls.__name__ and settings.right and not settings.ckpt:
        settings.ckpt = settings.right

    device = torch.device(settings.device)
    model.to(device)
    discriminators.to(device)
    model.discriminators = discriminators

    if settings.test:
        if not settings.headless:
            print("Setting up viewer for GUI mode...")
            env.render() # Setup viewer only if not headless
        else:
            if not settings.record:
                 print("Running test in headless mode WITHOUT recording.")
            else:
                 print(f"Running test in headless mode WITH recording to {settings.record}")

        # Pass total frames to test function only if recording and frames are specified
        total_frames_arg = settings.frames if settings.headless and settings.record and settings.frames is not None else None
        test(env, model, total_frames=total_frames_arg) # Pass frame limit to test function
    else:
        train(env, model, settings.ckpt, training_params)

    # <<< Explicitly close environment at the very end >>>
    # Note: test() and train() already have finally blocks with env.close()
    # This is an extra safeguard in case the script exits differently.
    print("Script finished. Ensuring environment cleanup...")
    if "env" in locals() and hasattr(env, "close") and callable(getattr(env, "close")):
        # Check if it might have already been closed by train/test
        if (
            hasattr(env, "sim") and env.sim is not None
        ):  # Heuristic: check if sim still exists
            env.close()
        else:
            print("Environment seems already closed.")
    else:
        print(
            "Environment variable 'env' not found or has no close method at script end."
        )