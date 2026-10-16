from __future__ import annotations

import logging
import time
import requests

logger = logging.getLogger(__name__)


def wait_lora_patch(
    max_wait: int = 300,
    retry_interval: float = 2.0,
    initial_delay: float = 1.0,
) -> None:
    """Retry VLLMClient.load_lora_adapter across FUSE mount propagation delays."""
    from trl.experimental.async_grpo.vllm_client import VLLMClient

    if getattr(VLLMClient, "_wait_lora_patched", False):
        return

    def retrying_load_lora_adapter(self, lora_name: str, lora_path: str, timeout: int = 1800) -> None:
        start = time.time()
        if initial_delay > 0:
            logger.info(
                f"[wait_lora_patch] Waiting {initial_delay:.1f}s for remote mount flush before loading '{lora_name}'..."
            )
            time.sleep(initial_delay)

        attempt = 0
        while True:
            attempt += 1
            url = f"{self.server_url}/v1/load_lora_adapter"
            payload = {"lora_name": lora_name, "lora_path": lora_path, "load_inplace": False}

            try:
                response = requests.post(url, json=payload, timeout=timeout)
            except Exception as e:
                elapsed = time.time() - start
                if elapsed < max_wait:
                    logger.warning(
                        f"[wait_lora_patch] Connection error calling {url} (attempt {attempt}, elapsed {elapsed:.1f}s): {e}. "
                        f"Retrying in {retry_interval}s..."
                    )
                    time.sleep(retry_interval)
                    continue
                raise

            elapsed = time.time() - start

            if response.status_code == 200:
                if attempt > 1:
                    logger.info(
                        f"[wait_lora_patch] Loaded LoRA adapter '{lora_name}' after {attempt} attempts ({elapsed:.1f}s total)."
                    )
                else:
                    logger.info(f"[wait_lora_patch] Loaded LoRA adapter '{lora_name}' on first attempt ({elapsed:.1f}s total).")
                return

            resp_text = response.text

            if response.status_code == 400 and "already been loaded" in resp_text:
                logger.info(f"[wait_lora_patch] Adapter '{lora_name}' already loaded in vLLM. Unloading stale adapter first...")
                try:
                    self.unload_lora_adapter(lora_name)
                except Exception as unload_err:
                    logger.warning(f"[wait_lora_patch] Failed to unload stale adapter '{lora_name}': {unload_err}")
                continue

            if response.status_code in (404, 500) and "No adapter found" in resp_text:
                if elapsed < max_wait:
                    logger.info(
                        f"[wait_lora_patch] Received HTTP {response.status_code} ('No adapter found'). "
                        f"Adapter '{lora_name}' at '{lora_path}' not visible on vLLM mount yet "
                        f"(attempt {attempt}, elapsed {elapsed:.1f}s / max {max_wait}s). Retrying in {retry_interval}s..."
                    )
                    time.sleep(retry_interval)
                    continue
                else:
                    logger.error(
                        f"[wait_lora_patch] Timed out waiting for adapter '{lora_name}' at '{lora_path}' after {elapsed:.1f}s."
                    )
                    raise TimeoutError(
                        f"Timed out after {elapsed:.1f}s waiting for adapter '{lora_name}' at '{lora_path}' to appear on vLLM mount: {resp_text}"
                    )

            if response.status_code == 404:
                raise RuntimeError(
                    f"The vLLM server at {self.server_url} does not expose `/v1/load_lora_adapter`. "
                    f"Restart it with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` in its environment. Response: {resp_text}"
                )

            if elapsed < max_wait and response.status_code >= 500:
                logger.warning(
                    f"[wait_lora_patch] Server error {response.status_code}: {resp_text}. Retrying in {retry_interval}s..."
                )
                time.sleep(retry_interval)
                continue

            raise RuntimeError(f"Failed to load LoRA adapter '{lora_name}': HTTP {response.status_code} - {resp_text}")

    VLLMClient.load_lora_adapter = retrying_load_lora_adapter
    VLLMClient._wait_lora_patched = True
    logger.info("Applied wait_lora_patch to VLLMClient.load_lora_adapter.")
