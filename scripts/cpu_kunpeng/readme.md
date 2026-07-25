# Prefill Multi-Instance Guide

## Configuration

Configure two prefill instances in `env.sh`:

```bash
# Default instance (short prompts)
PREFILL_IP_SPEC="xxx.xxx.xxx. | 1-16"
PREFILL_MASTER_ADDR="xxx.xxx.xxx.1"
PREFILL_MASTER_PORT="5000"
PREFILL_PP_SIZE=1

# Long prompt instance
PREFILL_LONG_PROMPT_IP_SPEC="xxx.xxx.xxx. | 17-32"
PREFILL_LONG_PROMPT_MASTER_ADDR="xxx.xxx.xxx.17"
PREFILL_LONG_PROMPT_MASTER_PORT="5020"
PREFILL_LONG_PROMPT_PP_SIZE=2

# Bucket policy switch: 1=enable dual-instance + bucket load balancing, 0=single instance
export PREFILL_BUCKET=1
```

## Launch

### 1. Single-Instance Mode (Default)

```bash
# prefill (run on node 1)
sh launch.sh prefill

# decode (run on decode master node)
sh launch.sh decode

# router (run on router node)
sh launch.sh router
```

### 2. Dual-Instance Mode (Bucket Policy)

```bash
# Instance 1 (short-prompt prefill, nodes 1-16, run on node 1)
sh launch.sh prefill

# Instance 2 (long-prompt prefill, nodes 17-32, run on node 17)
PREFILL_INSTANCE=long_prompt sh launch.sh prefill

# decode (run on decode master node)
sh launch.sh decode

# router (run on router node)
export PREFILL_BUCKET=1 && sh launch.sh router
```

## Notes

- The two prefill instances have different master nodes; launch each on its respective master node
- The `MASTER_PORT` of the two instances must not be the same (default: 5000 vs 5020)
- When `PREFILL_BUCKET=1`, the router automatically registers both prefill instances and enables the bucket policy
- When `PREFILL_BUCKET=0`, the router registers only one prefill instance and uses the policy specified by `--policy`
- The `PP_SIZE` of the long-prompt instance can be configured independently (default: 2)
