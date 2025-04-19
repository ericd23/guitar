import os, time
import importlib
from collections import namedtuple
import argparse # Ensure argparse is imported

import env # Assuming env.py is in the same directory or python path
from models import ACModel, Discriminator, AdaptNet
from utils import seed

import torch
import numpy as np

# <<< START MODIFIED ARGUMENT PARSER >>>
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
    help="Specific music note policy training or evaluation.")

parser.add_argument("--left", type=str, default=None,
    help="Checkpoint directory or file for left-hand policy training or evaluation.")
parser.add_argument("--right", type=str, default=None,
    help="Checkpoint directory or file for right-hand policy training or evaluation.")

# --- New/Modified Arguments for Recording ---
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
# --- End New/Modified Arguments ---

settings = parser.parse_args()
# <<< END MODIFIED ARGUMENT PARSER >>>


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

# <<< START MODIFIED test() FUNCTION >>>
def test(env, model, total_frames=None): # Add total_frames parameter
    model.eval()
    env.eval()
    env.reset()
    # Initialize lists to store metrics *since last print*
    accuracy_l, precision_l, recall_l = [], [], []
    accuracy_r, precision_r, recall_r = [], [], []
    metrics_ready_to_print = False # Flag to print on next iteration after collecting
    nn = 0 # Note/Reset counter
    frame_count = 0 # Initialize frame counter

    print("Starting evaluation loop...")
    if total_frames:
        print(f"Will stop after {total_frames} frames.")

    try: # Use try/finally to ensure env.close() is called
        while not env.request_quit:
            # Print metrics collected in the *previous* step before getting new observation
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

            obs_, rews, dones, info = env.step(actions) # dones here is the *new* done status

            # --- Metric Collection ---
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
            # --- End Metric Collection ---

            frame_count += 1

            # --- Loop Termination Condition ---
            if total_frames is not None and frame_count >= total_frames:
                print(f"\nReached target frame count: {frame_count}/{total_frames}. Stopping.")
                break # Exit the loop
            # --- End Loop Termination Condition ---

    finally: # Ensure cleanup happens
        print("Evaluation loop finished or interrupted.")
        # Check if env exists and has close method before calling
        if 'env' in locals() and hasattr(env, 'close') and callable(getattr(env, 'close')):
            print("Closing environment and releasing resources...")
            env.close()
        else:
             print("Environment does not exist or does not have a close method.")
# <<< END MODIFIED test() FUNCTION >>>


def train(env, model, ckpt_dir, training_params):
    # --- Keep original train() function ---
    # (Consider adding a try/finally block with env.close() here too for robustness)
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

    accuracy_l, precision_l, recall_l = [], [], []
    accuracy_r, precision_r, recall_r = [], [], []
    try: # Add try/finally for cleanup
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
                if "accuracy_r" in info and info["accuracy_r"].numel() > 0:
                    accuracy_r.append(info["accuracy_r"])
                    precision_r.append(info["precision_r"])
                    recall_r.append(info["recall_r"])

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
                            ).unsqueeze_(0)      # 1 x L
                            mask = length <= end_frame.unsqueeze(1)      # N x L

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

                    for batch in range(n_samples//BATCH_SIZE):
                        sample = idx[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                        for name, disc, ref, ob, seq_end_frame_ in disc_data_training:

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

                model.eval()
                for v in buffer.values(): v.clear()
                for buf in buffer_disc.values():
                    for v in buf.values(): v.clear()

                if epoch % LOG_INTERVAL == 1:
                    acc_l_val = torch.mean(torch.cat(accuracy_l)).item() if accuracy_l else np.nan
                    prec_l_val = torch.mean(torch.cat(precision_l)).item() if precision_l else np.nan
                    rec_l_val = torch.mean(torch.cat(recall_l)).item() if recall_l else np.nan

                    acc_r_val = torch.mean(torch.cat(accuracy_r)).item() if accuracy_r else np.nan
                    prec_r_val = torch.mean(torch.cat(precision_r)).item() if precision_r else np.nan
                    rec_r_val = torch.mean(torch.cat(recall_r)).item() if recall_r else np.nan

                    if not settings.silent:
                        print("Epoch: {}, L(Ac/Pr/Re): {:.4f}/{:.4f}/{:.4f}, R(Ac/Pr/Re): {:.4f}/{:.4f}/{:.4f} -- {:.4f}s".format(
                            epoch, acc_l_val, prec_l_val, rec_l_val, acc_r_val, prec_r_val, rec_r_val, time.time()-tic
                        ))
                    if logger is not None:
                        # Log metrics
                        if accuracy_l:
                            logger.add_scalar("train/l_accuracy", acc_l_val, epoch)
                            logger.add_scalar("train/l_precision", prec_l_val, epoch)
                            logger.add_scalar("train/l_recall", rec_l_val, epoch)
                        if accuracy_r:
                            logger.add_scalar("train/r_accuracy", acc_r_val, epoch)
                            logger.add_scalar("train/r_precision", prec_r_val, epoch)
                            logger.add_scalar("train/r_recall", rec_r_val, epoch)
                        # Optional: Log other metrics like reward components if needed

                    accuracy_l.clear(); precision_l.clear(); recall_l.clear()
                    accuracy_r.clear(); precision_r.clear(); recall_r.clear()

                if ckpt_dir is not None:
                    state = None
                    if epoch % 500 == 0:
                        state = dict(model=model.state_dict())
                        save_path = os.path.join(ckpt_dir, "ckpt")
                        torch.save(state, save_path)
                        print(f"Saved checkpoint: {save_path}")
                    if epoch % training_params.save_interval == 0:
                        if state is None:
                            state = dict(model=model.state_dict())
                        save_path = os.path.join(ckpt_dir, f"ckpt-{epoch}")
                        torch.save(state, save_path)
                        print(f"Saved checkpoint: {save_path}")
                if epoch >= training_params.max_epochs:
                     print("Reached max epochs. Exiting training.")
                     break # Exit training loop
                tic = time.time()
    finally: # Ensure cleanup in train mode too
        print("Training loop finished or interrupted.")
        if 'env' in locals() and hasattr(env, 'close') and callable(getattr(env, 'close')):
            print("Closing environment and releasing resources...")
            env.close()
        else:
             print("Environment does not exist or does not have a close method.")


if __name__ == "__main__":
    if os.path.splitext(settings.config)[-1] in [".pkl", ".json", ".yaml"]:
        config = object() # Create dummy config object
        # Basic default structure expected by the instantiation logic
        config.env_params = dict(
            motion_file = settings.config
        )
        config.env_cls = 'ICCGANHumanoid' # Default if config is not a .py file
        config.discriminators = {}
        config.seed = settings.seed
        print(f"Warning: Config file {settings.config} is not a .py file. Using default structure.")
    else:
        spec = importlib.util.spec_from_file_location("config", settings.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

    seed(config.seed if hasattr(config, "seed") else settings.seed)

    if hasattr(config, "training_params"):
        TRAINING_PARAMS.update(config.training_params)
    # Ensure save_interval is sensible
    if not TRAINING_PARAMS["save_interval"] or TRAINING_PARAMS["save_interval"] > TRAINING_PARAMS["max_epochs"]:
        TRAINING_PARAMS["save_interval"] = TRAINING_PARAMS["max_epochs"]
    print("Effective Training Params:", TRAINING_PARAMS)
    training_params = namedtuple('x', TRAINING_PARAMS.keys())(*TRAINING_PARAMS.values())


    # <<< START MODIFIED ENVIRONMENT INSTANTIATION (main block) >>>
    # (Copied from above for completeness in the final block)
    # --- Environment Instantiation ---
    env_params_from_config = config.env_params if hasattr(config, "env_params") else {}
    env_kwargs = env_params_from_config.copy() # Start with config params

    # Check for discriminators in config
    discriminators_cfg = {} # Use a different name to avoid conflict
    if hasattr(config, "discriminators"):
         discriminators_cfg = {
               name: env.DiscriminatorConfig(**prop)
               for name, prop in config.discriminators.items()
         }

    # Determine env_cls
    if hasattr(config, "env_cls"):
        env_cls = getattr(env, config.env_cls)
    else:
        env_cls = env.ICCGANHumanoid # Example Default (adjust if needed)

    # Set FPS from command line directly. It affects sim dt and video recording.
    env_kwargs["fps"] = settings.fps

    # Add headless/recording specific params (Env/HeadlessEnv init will use them if needed)
    env_kwargs["record"] = settings.record
    env_kwargs["cam_width"] = settings.width
    env_kwargs["cam_height"] = settings.height

    # Handle --note argument
    if settings.note is not None:
        env_kwargs["note_file"] = settings.note
        if settings.test:
            # Ensure sequential playback in test mode when a note file is given
            env_kwargs["random_note_sampling"] = False

    # Set number of environments and episode length based on test mode
    if settings.test:
        num_envs = 1
        # Use --frames for episode length if recording, otherwise a large default
        if settings.record and settings.frames:
             env_kwargs["episode_length"] = settings.frames
        else:
             env_kwargs["episode_length"] = 500000 # Large default if not recording or no frame limit
    else:
        num_envs = training_params.num_envs
        # Keep the check for existing checkpoint during training
        if settings.ckpt and not settings.test and (os.path.isfile(settings.ckpt) or os.path.exists(os.path.join(settings.ckpt, "ckpt"))):
            raise ValueError("Checkpoint folder {} exists. Add `--test` option to run test or remove checkpoint.".format(settings.ckpt))

    print(f"Instantiating environment: {env_cls.__name__}")
    # print(f"Environment Kwargs: {env_kwargs}") # Optional: for debugging

    env = env_cls(n_envs=num_envs, # Use n_envs consistently
                  discriminators=discriminators_cfg, # Pass parsed discriminators
                  compute_device=settings.device,
                  # Ensure graphics device matches headless state
                  graphics_device=settings.device if settings.headless else None,
                  **env_kwargs # Pass all collected kwargs
                 )
    # --- End Environment Instantiation ---
    # <<< END MODIFIED ENVIRONMENT INSTANTIATION (main block) >>>


    # --- Model Instantiation and Loading ---
    value_dim = len(env.discriminators)+env.rew_dim if hasattr(env, 'discriminators') else env.rew_dim # Handle if no discriminators
    goal_dim_tuple = env.goal_dim if isinstance(env.goal_dim, (list, tuple)) else (env.goal_dim, env.goal_dim) # Ensure tuple

    model = ACModel(env.state_dim, env.act_dim, goal_dim_tuple, value_dim)
    discriminators_module = torch.nn.ModuleDict({
        name: Discriminator(dim) for name, dim in env.disc_dim.items()
    }) if hasattr(env, 'disc_dim') else torch.nn.ModuleDict() # Handle if no discriminators

    # --- AdaptNet Logic (Handling Potential Errors) ---
    if "Two" in env_cls.__name__:
        if not ((settings.left and settings.right) or settings.ckpt):
             print("Warning: TwoHands environment selected, but neither --ckpt nor both --left and --right provided. Model may not be properly initialized.")
             # Decide behavior: continue with random weights, or raise error?
             # raise ValueError("Must provide --ckpt or both --left and --right for TwoHands environment.")

        state_dict_two = None
        if settings.ckpt:
            ckpt_path_two = None
            if os.path.isdir(settings.ckpt):
                ckpt_path_two = os.path.join(settings.ckpt, "ckpt")
            else:
                ckpt_path_two = settings.ckpt
            if ckpt_path_two and os.path.exists(ckpt_path_two):
                 try:
                      state_dict_two = torch.load(ckpt_path_two, map_location=torch.device(settings.device))
                      if 'model' in state_dict_two: state_dict_two = state_dict_two['model'] # Use nested dict if present
                      print(f"Loaded combined state dict from {ckpt_path_two}")
                 except Exception as e:
                      print(f"Error loading combined checkpoint {ckpt_path_two}: {e}. AdaptNet might not initialize correctly.")
                      state_dict_two = None # Reset on error
            else:
                 print(f"Warning: Combined checkpoint {ckpt_path_two} not found.")


        # Initialize dummy policies, dimensions will be inferred if state_dict_two loaded
        left_policy, right_policy = None, None
        default_state_dim, default_act_dim, default_goal_dim, default_latent_dim = 10, 5, 5, 256 # Example defaults

        try: # Wrap policy creation in try block
            # Try loading left policy
            state_dict_left = None
            if settings.left:
                ckpt_path_left = os.path.join(settings.left, "ckpt") if os.path.isdir(settings.left) else settings.left
                if os.path.exists(ckpt_path_left):
                     state_dict_left = torch.load(ckpt_path_left, map_location=torch.device(settings.device))
                     if 'model' in state_dict_left: state_dict_left = state_dict_left['model']
                     print(f"Loaded left state dict from {ckpt_path_left}")
                else: print(f"Warning: Left checkpoint {ckpt_path_left} not found.")

            # Infer dimensions or use defaults for left
            if state_dict_left:
                state_dim_l = state_dict_left["ob_normalizer.mean"].shape[0]
                act_dim_l = state_dict_left["actor.mu.bias"].shape[0]
                goal_dim_l = state_dict_left["actor.embed_goal.0.weight"].shape[1]
                latent_dim_l = state_dict_left["actor.mlp.0.weight"].shape[1]
            elif state_dict_two and "actor.meta1.ob_normalizer.mean" in state_dict_two: # Infer from combined if possible
                state_dim_l = state_dict_two["actor.meta1.ob_normalizer.mean"].shape[0]
                act_dim_l = state_dict_two["actor.meta1.mu.bias"].shape[0]
                goal_dim_l = state_dict_two["actor.meta1.embed_goal.0.weight"].shape[1]
                latent_dim_l = state_dict_two["actor.meta1.mlp.0.weight"].shape[1]
            else:
                 state_dim_l, act_dim_l, goal_dim_l, latent_dim_l = default_state_dim, default_act_dim, default_goal_dim, default_latent_dim
                 print("Warning: Using default dimensions for left policy.")

            left_policy = ACModel.Actor(state_dim_l, act_dim_l, goal_dim_l, latent_dim_l)
            left_policy.ob_normalizer = env.RunningMeanStd(state_dim_l, clamp=5.0) # Create normalizer instance
            if state_dict_left: # Load weights if individual ckpt provided
                policy_state_l, ob_state_l = {}, {}
                for k, v in state_dict_left.items():
                    if k.startswith("actor."): policy_state_l[k[6:]] = v
                    if k.startswith("ob_normalizer."): ob_state_l[k[14:]] = v
                left_policy.load_state_dict(policy_state_l, strict=False)
                left_policy.ob_normalizer.load_state_dict(ob_state_l, strict=False)

            # Try loading right policy (similar logic)
            state_dict_right = None
            if settings.right:
                ckpt_path_right = os.path.join(settings.right, "ckpt") if os.path.isdir(settings.right) else settings.right
                if os.path.exists(ckpt_path_right):
                     state_dict_right = torch.load(ckpt_path_right, map_location=torch.device(settings.device))
                     if 'model' in state_dict_right: state_dict_right = state_dict_right['model']
                     print(f"Loaded right state dict from {ckpt_path_right}")
                else: print(f"Warning: Right checkpoint {ckpt_path_right} not found.")

            if state_dict_right:
                state_dim_r = state_dict_right["ob_normalizer.mean"].shape[0]
                act_dim_r = state_dict_right["actor.mu.bias"].shape[0]
                goal_dim_r = state_dict_right["actor.embed_goal.0.weight"].shape[1]
                latent_dim_r = state_dict_right["actor.mlp.0.weight"].shape[1]
            elif state_dict_two and "actor.meta2.ob_normalizer.mean" in state_dict_two:
                state_dim_r = state_dict_two["actor.meta2.ob_normalizer.mean"].shape[0]
                act_dim_r = state_dict_two["actor.meta2.mu.bias"].shape[0]
                goal_dim_r = state_dict_two["actor.meta2.embed_goal.0.weight"].shape[1]
                latent_dim_r = state_dict_two["actor.meta2.mlp.0.weight"].shape[1]
            else:
                 state_dim_r, act_dim_r, goal_dim_r, latent_dim_r = default_state_dim, default_act_dim, default_goal_dim, default_latent_dim
                 print("Warning: Using default dimensions for right policy.")

            right_policy = ACModel.Actor(state_dim_r, act_dim_r, goal_dim_r, latent_dim_r)
            right_policy.ob_normalizer = env.RunningMeanStd(state_dim_r, clamp=5.0) # Create normalizer instance
            if state_dict_right: # Load weights if individual ckpt provided
                 policy_state_r, ob_state_r = {}, {}
                 for k, v in state_dict_right.items():
                     if k.startswith("actor."): policy_state_r[k[6:]] = v
                     if k.startswith("ob_normalizer."): ob_state_r[k[14:]] = v
                 right_policy.load_state_dict(policy_state_r, strict=False)
                 right_policy.ob_normalizer.load_state_dict(ob_state_r, strict=False)

            # Wrap with AdaptNet if both policies were created
            if left_policy and right_policy:
                 print("Wrapping model actor with AdaptNet.")
                 model.actor = AdaptNet(model, left_policy, right_policy) # Pass the main model too
                 if env.reward_weights is not None: env.reward_weights *= 2 # Adjust reward weights as before
            else:
                 print("Warning: Could not create both left and right policies for AdaptNet.")

        except Exception as e:
             print(f"Error during AdaptNet policy creation or dimension inference: {e}. Continuing without AdaptNet wrapping.")


    elif "Left" in env_cls.__name__ and settings.left and not settings.ckpt:
        settings.ckpt = settings.left
    elif "Right" in env_cls.__name__ and settings.right and not settings.ckpt:
        settings.ckpt = settings.right
    # --- End AdaptNet Logic ---

    device = torch.device(settings.device)
    model.to(device)
    discriminators_module.to(device)
    model.discriminators = discriminators_module


    # --- Final Checkpoint Loading (for combined model or single hand) ---
    ckpt_path_to_load = None
    if settings.ckpt:
         if os.path.isdir(settings.ckpt):
              ckpt_path_to_load = os.path.join(settings.ckpt, "ckpt")
         else:
              ckpt_path_to_load = settings.ckpt

         if ckpt_path_to_load and os.path.exists(ckpt_path_to_load):
              print(f"Load final model state from {ckpt_path_to_load}")
              try:
                   # state_dict = torch.load(ckpt_path_to_load, map_location=torch.device(settings.device))
                   # Load the previously loaded dict if AdaptNet handled it, otherwise load fresh
                   final_state_dict = state_dict_two if state_dict_two else torch.load(ckpt_path_to_load, map_location=device)
                   if 'model' in final_state_dict: final_state_dict = final_state_dict['model'] # Handle nesting again

                   model.load_state_dict(final_state_dict, strict=False)
                   print("Final model state loaded successfully.")
              except Exception as e:
                   print(f"Error loading final model state from {ckpt_path_to_load}: {e}")
         elif settings.test: # Error only if testing and checkpoint is missing
              raise FileNotFoundError(f"Checkpoint not found at {ckpt_path_to_load}, required for --test mode.")
         else: # Warning if training and checkpoint is missing
              print(f"Warning: Checkpoint not found at {ckpt_path_to_load}. Starting training from scratch.")
    elif settings.test:
         raise ValueError("Must provide --ckpt for --test mode.")
    # --- End Final Checkpoint Loading ---

    # --- Main Execution ---
    if settings.test:
        if not settings.headless:
            print("Setting up viewer...")
            env.render() # Setup viewer only if not headless
        # Pass total frames to test function only if recording and frames are specified
        total_frames_arg = settings.frames if settings.record and settings.frames is not None else None
        test(env, model, total_frames=total_frames_arg)
    else:
        # Create checkpoint directory if it doesn't exist for training
        if settings.ckpt and not os.path.exists(settings.ckpt):
             print(f"Creating checkpoint directory: {settings.ckpt}")
             os.makedirs(settings.ckpt)
        train(env, model, settings.ckpt, training_params)
    # --- End Main Execution ---