"""OpenPI server adapter with inference RNG paired by episode and query."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path


class PairedPolicy:
    def __init__(self, policy):
        if not hasattr(policy, "_rng"):
            raise RuntimeError("paired RNG adapter requires the inspected JAX OpenPI policy")
        self.policy = policy

    def infer(self, obs):
        import jax
        request = dict(obs)
        seed = request.pop("_sc_inference_seed", None)
        query = request.pop("_sc_query_index", None)
        if type(seed) is not int or type(query) is not int or not 0 <= seed < 2**32 or not 0 <= query < 2**32:
            raise ValueError("paired inference requires non-negative uint32 seed/query")
        self.policy._rng = jax.random.fold_in(jax.random.key(seed), query)
        return self.policy.infer(request)


def main():
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.training.checkpoints import load_norm_stats
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    norm_path = Path(args.checkpoint) / "assets" / "lerobot" / "norm_stats.json"
    if not norm_path.is_file():
        raise RuntimeError(f"training normalization artifact missing: {norm_path}")
    norm_stats = load_norm_stats(Path(args.checkpoint) / "assets", "lerobot")
    policy = policy_config.create_trained_policy(config.get_config(args.config), args.checkpoint, norm_stats=norm_stats)
    server = WebsocketPolicyServer(PairedPolicy(policy), host="127.0.0.1", port=args.port, metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
