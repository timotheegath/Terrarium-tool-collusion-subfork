# Terrarium for agentic collusion via tools

This is a fork of Terrarium focusing on the hospital environment as part of the evaluation of agentic collusion using tools.
The main contributor will primarily use this fork as a submodule of [the Colosseum-style evaluation repo](https://github.com/timotheegath/collusion-via-tools).
Terrarium is an open-source framework for building controlled multi-agent environments and testing how communication, topology, private information, tools, model providers, and adversarial conditions affect coordination.

It is not primarily a production agent orchestrator or a fixed benchmark. Terrarium is built for researchers, engineers, and students who want to run repeatable multi-agent experiments where the environment, communication structure, and attack surface are explicit.

## Why the fork?

Throughout the first trial of setting up this evaluation, we needed to make several fundamental changes to the environment itself. Those were rapidly done as monkey patches. However:

- This hurt the code visibility and clarity
- These changes would not be easily portable to the main Terrarium repo if they happened to be useful to it
- The parent repo, which is itself a fork of the Colosseum repo, was still depending on an old version of Terrarium.

In the spirit of readability and potential contribution, I decided to include this fork as a submodule of the parent repo, instead of monkey-patching it.

## Install

For local development:

```bash
git clone https://github.com/timotheegath/Terrarium-tool-collusion-subfork
cd Terrarium
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync
```

Some DCOP environments use CoLLAB instance generation. If CoLLAB is not available through the repository submodule, clone it and point Terrarium at it:

```bash
git clone https://github.com/Saad-Mahmud/CoLLAB_SEA.git /path/to/CoLLAB
export TERRARIUM_COLLAB_PATH=/path/to/CoLLAB
```

## First Run

Copy the environment template and set only the provider keys you need:

```bash
cp -n .env.example .env
```

Then run a meeting-scheduling simulation:

```bash
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml
```

Terrarium writes prompts, tool calls, blackboard state, communication network plots, and agent trajectories below:

```text
logs/<environment>/<tag_model>/<run_timestamp>/seed_<seed>/
```

## Minimal Config

Terrarium runs are configured with YAML. The four required sections are `simulation`, `environment`, `communication_network`, and `llm`.

```yaml
simulation:
  max_iterations: 3
  max_planning_rounds: 1
  max_conversation_steps: 3
  seed: 42
  tags:
    - baseline

environment:
  name: MeetingSchedulingEnvironment
  assignment_filling: false
  num_meetings: 2
  timeline_length: 12
  min_participants: 2
  max_participants: 4

communication_network:
  topology: erdos_renyi
  num_agents: 2
  edge_prob: 0.7
  consolidate_channels: true

llm:
  provider: openai
  openai:
    model: gpt-4.1-mini
    params:
      max_tokens: 1024
      temperature: 0.7
```

See the [Quick Start](https://aisec.cs.umass.edu/Terrarium/latest/quickstart.html) for the full configuration guide.

## Core Ideas

Terrarium runs are built from components. Swap one component while holding the others fixed to isolate the effect of an environment, protocol, topology, model, or attack.

| Component | Base surface | What it controls |
| --- | --- | --- |
| Agents | `BaseAgent` | LLM/tool loop for each participant. |
| Environments | `AbstractEnvironment` | Task state, private observations, legal actions, rewards, and stopping conditions. |
| Communication protocols | `BaseCommunicationProtocol` | Turn order, phases, and tool routing. |
| Communication Networks | `CommunicationNetwork` | Which agents can share communication channels. |
| Communication Channels | `Blackboard` / `Megaboard` | Append-only message state visible to channel participants. |
| Tools | `ToolsetDiscovery` and environment `Tools` classes | Blackboard communication, environment actions, and optional external MCP tools. |
| LLM clients | `AbstractClient` | Provider-specific generation and tool-call parsing. |

Component guides live in the [hosted documentation](https://aisec.cs.umass.edu/Terrarium/latest/components/environments.html).

## Repository Layout

```text
terrarium/
  agents/                  # Base agent and factory
  attacks/                 # Reference adversarial components
  communication_protocols/ # Planning/execution protocols
  core/                    # Blackboards, loggers, async helpers
  environments/            # Implemented environments and task tools
  llm/                     # Provider clients and vLLM runtime
  networks/                # Communication topology builders
  tools/                   # Tool discovery and prompt helpers

examples/
  base_main.py             # Standard simulation runner
  attack_main.py           # Attack simulation runner
  configs/                 # Example YAML configs

docs/                      # Documentation source
dashboards/                # Static dashboard utilities
```

## Contributing

Contributions are welcome, especially focused fixes, new environments, new tools, documentation improvements, and reproducible attack or evaluation workflows.

For larger changes, open an issue first so the scope is clear. A good pull request should include:

- A focused description of the change and why it is needed.
- Updated docs or examples when behavior changes.
- Relevant checks, or a short note explaining why a check was not run.
- A small config or reproduction when the change affects simulation behavior.

## Citation

If you use Terrarium in your research, please cite us using the following:

```bibtex
@article{nakamura2025terrarium,
  title={Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies},
  author={Nakamura, Mason and Kumar, Abhinav and Mahmud, Saaduddin and Abdelnabi, Sahar and Zilberstein, Shlomo and Bagdasarian, Eugene},
  journal={arXiv preprint arXiv:2510.14312},
  year={2025}
}
```

## License

Terrarium is released under the MIT License. See [LICENSE](LICENSE).
