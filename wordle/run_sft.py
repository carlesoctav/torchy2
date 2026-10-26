from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parent
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))

import sws
from torchy_core.infra import ColabSession, ProcessGroup
from torchy_core.sws_utils import extract_config_and_argv, load_config_builder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")

def build_trainer_command(config_path: str, *other_argv: str) -> list[str]:
    return [
        "uv", "run", "python", "sft.py",
        "--config", config_path,
        *other_argv,
    ]


def main(cfg: sws.FinalConfig, unused_argv: list[str] | None = None):

    parser = argparse.ArgumentParser(description="run simple sft")
    parser.add_argument("--stop-vm", action="store_true", help="Stop Colab VMs upon completion or interrupt")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing Colab sessions without recreating them")
    parser.add_argument("--gpu", default=None, help="Override GPU accelerator (e.g. t4, l4, a100)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting VMs")
    extra_args, _ = parser.parse_known_args(unused_argv)

    train_infra_config = cfg.train.infra.to_dict()
    project_name = cfg.project_name

    cfg_path, other_argv = extract_config_and_argv(sys.argv[1:])

    remote_root = cfg.remote_root.rstrip("/")
    remote_exp_dir = f"{remote_root}/{exp_dir.name}"
    sync_dirs = {
        str(repo_root): remote_root,
    }

    if extra_args.reuse:
        train_infra_config["reuse"] = True
    if extra_args.stop_vm:
        train_infra_config["stop_vm"] = True
    if extra_args.gpu:
        train_infra_config["gpu"] = extra_args.gpu

    logger.info(f"Config path: {cfg_path}")
    logger.info(f"Experiment: {cfg.exp_name} | Project: {cfg.project_name}")
    logger.info(f"Sync mapping: {sync_dirs}")
    trainer_cmd = build_trainer_command(cfg_path, *other_argv)

    if extra_args.dry_run:
        logger.info("[Dry Run] Trainer command: " + " ".join(trainer_cmd))
        return

    log_dir = exp_dir / "logs" / f"{project_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    trainer_log = log_dir / f"{cfg.exp_name}_trainer.log"
    trainer_log.unlink(missing_ok=True)

    trainer_session = ColabSession(**train_infra_config)

    with ProcessGroup() as pg:

        trainer_proc = trainer_session.Popen(
            cmd=trainer_cmd,
            log_path=trainer_log,
            dir=sync_dirs,
            work_dir=remote_exp_dir,
        )
        pg.add("trainer", trainer_proc, session=trainer_session)

        pg.wait("trainer")
        logger.info("Trainer finished.")


if __name__ == "__main__":
    sws.run(main, forward_extras=True)

