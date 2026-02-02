set shell := ["bash", "-euo", "pipefail", "-c"]

prime_host := env_var_or_default("PRIME_HOST", "prime-node")
remote_root := env_var_or_default("REMOTE_ROOT", "~/MegaDLMs")

bootstrap-remote:
	ssh {{prime_host}} 'bash -s' < scripts/bootstrap_remote.sh

data-remote:
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/download_simplestories_data.sh"

train script="examples/dlm_training/dlm_pretrain_simplestories_8k.sh":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_train_remote.sh {{script}}"

sweep-train config="examples/dlm_training/dlm_pretrain_simplestories_8k_sweep.yaml" extra="":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_sweep_train_remote.sh $(printf '%q' '{{config}}') $(printf '%q' '{{extra}}')"

attach-train:
	ssh -t {{prime_host}} 'tmux attach -t megadlms-train'

attach-sweep:
	ssh -t {{prime_host}} 'tmux attach -t megadlms-sweep-train'

kill-train:
	ssh {{prime_host}} 'tmux kill-session -t megadlms-train 2>/dev/null || true'

kill-sweep:
	ssh {{prime_host}} 'tmux kill-session -t megadlms-sweep-train 2>/dev/null || true'

nvitop:
	ssh -t {{prime_host}} 'export PATH="$HOME/.local/bin:$PATH"; uvx nvitop'

sync-env:
	if [ ! -f env/.env ]; then echo "Missing env/.env" >&2; exit 1; fi
	ssh {{prime_host}} "mkdir -p {{remote_root}}/env"
	scp env/.env {{prime_host}}:{{remote_root}}/env/.env

auto-train: bootstrap-remote data-remote sync-env train

auto-sweep-train: bootstrap-remote data-remote sync-env sweep-train
