from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def ensure_bucket(bucket_id: str) -> None:
    from trackio.bucket_storage import create_bucket_if_not_exists

    create_bucket_if_not_exists(bucket_id)


def project_dir_for(output_dir: str) -> str:
    resolved = Path(output_dir).resolve()
    if resolved.parent == Path.cwd().resolve():
        return str(resolved)
    return str(resolved.parent)


class BucketSyncCallback(TrainerCallback):
    def __init__(self, config: Any):
        self.bucket_id = config.get("bucket_id") if hasattr(config, "get") else getattr(config, "bucket_id", None)
        self.enabled = bool(self.bucket_id)
        self._local_dir: str | None = None
        self._shut_down = False
        if self.enabled:
            try:
                ensure_bucket(self.bucket_id)
            except Exception as exc:
                warnings.warn(
                    f"BucketSyncCallback: cannot ensure bucket "
                    f"{self.bucket_id!r}: {exc}. Syncing disabled."
                )
                self.enabled = False

    @property
    def bucket_url(self) -> str:
        return f"hf://buckets/{self.bucket_id}"

    def _sync(self) -> None:
        if not self.enabled or self._local_dir is None:
            return
        from huggingface_hub import HfApi

        try:
            HfApi().sync_bucket(
                f"{self._local_dir}",
                self.bucket_url,
                exclude=["completions/*", "*/completions/*"],
            )
        except Exception as exc:
            warnings.warn(f"BucketSyncCallback: sync failed, will retry: {exc}")

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._local_dir is None:
            self._local_dir = project_dir_for(args.output_dir)
        self._sync()

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._local_dir is None:
            self._local_dir = project_dir_for(args.output_dir)
        self.shutdown()

    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        self._sync()
