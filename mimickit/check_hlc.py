import sys
import torch

import run                         
import util.mp_util as mp_util
from util.logger import Logger

def main(argv):
    args = run.load_args(argv)
    device = args.parse_strings("devices", ["cuda:0"])[0]
    num_envs = args.parse_int("num_envs", 4)

    mp_util.init(0, 1, device, 6000)
    run.set_rand_seed(args)

    env = run.build_env(args, num_envs, device, visualize=False)
    agent = run.build_agent(args, env, device) 

    # Check 1: slicing the controller's observation
    obs, info = env.reset()
    llc_obs = agent._extract_llc_obs(obs)
    Logger.print("full obs {}  ->  llc obs {}".format(tuple(obs.shape), tuple(llc_obs.shape)))
    assert llc_obs.shape[-1] == agent._llc_obs_dim

    # Check 2: the frozen controller turns (obs, random skill) into joint targets
    llc = agent._llc_agent
    with torch.no_grad():
        z = torch.nn.functional.normalize(torch.randn(num_envs, agent._latent_dim, device=device), dim=-1)
        norm_obs = llc._obs_norm.normalize(llc_obs)
        a = llc._a_norm.unnormalize(llc._model.eval_actor(norm_obs, z).mode)
    Logger.print("llc action {}, finite: {}".format(tuple(a.shape), torch.isfinite(a).all().item()))
    assert a.shape[-1] == env.get_action_space().shape[-1]

    # Check 3: the high-level agent's action side
    Logger.print("hlc action normalizer mean {:.3f}, std {:.3f}".format(
        agent._a_norm.get_mean().abs().max().item(), agent._a_norm.get_std().mean().item()))

    Logger.print("All checks passed.")

if __name__ == "__main__":
    main(sys.argv)