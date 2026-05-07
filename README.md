<div align="center">

# T²-GRPO - A Turn-Trajectory Group Relative Policy Optimization

### Environment-coherent multi-turn RL for dementia-care dialogue agents

T²-GRPO is a research training stack and testbed for caregiver policies
that interact with a frozen dementia-patient simulator and learn from both
trajectory-level outcomes and simulator-native turn-level behavioral signals.

[Method draft](paper/final_rps_method_section_locked_zh_9.md) ·
[Proposal](PROPOSAL.md) ·
[Engineering plan](ENGINEERING_PLAN.md) ·
[verl notes](docs/verl_integration.md) ·
[DemMA checkpoint](https://huggingface.co/hulehule/DemMA-Planner-SFT)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](#license)
[![Status](https://img.shields.io/badge/status-research%20preview-orange.svg)](#status)

</div>

---

## What Is T²-GRPO?

T²-GRPO targets a narrow but difficult RL setting: multi-turn dementia-care
conversations where a patient sincerely states something that conflicts with
reality, such as wanting to visit a deceased parent or insisting medication has
already been taken. The caregiver must respond over several turns while balancing
comfort, autonomy, factual discipline, and safety.

The core idea is to treat DemMA's inline patient-behavior annotations as an
environment-native turn reward channel. Instead of paying for an extra LLM judge
at every turn, the patient simulator already emits structured motion, facial,
and sound labels along with each utterance. Caregiver-R1 maps those labels to
clinical ordinal tiers and uses tier deltas as turn-level credit.

In short: **DemMA provides the patient environment; EC-MTRL provides the
dual-horizon reward and advantage estimator; verl/GDPO provides the scalable RL
training path.**


<a id="citation"></a>

```bibtex
@misc{caregiver_r1_2026,
  title  = {T²-GRPO},
  author = {T²-GRPO Team},
  year   = {2026},
  note   = {Research preview; preprint forthcoming.}
}
```

If you use the patient simulator, please also cite DemMA:

```bibtex
@inproceedings{demma2025,
  title     = {DemMA: A Multi-Agent Dementia-Patient Simulator for
               Caregiver Conversation Research},
  author    = {Hu et al.},
  booktitle = {Proceedings of ACL},
  year      = {2025}
}
```

## Acknowledgements

This project builds on
[verl](https://github.com/volcengine/verl), and GDPO-style group-relative
optimization. The clinical strategy menu draws on caregiver communication
frameworks including NURSE, VERA, SPIKES, DICE, Reality Orientation,
Therapeutic Fibbing, Reminiscence Therapy, Montessori methods, Redirection, and
Non-committal response.

## License

<a id="license"></a>

This repository is released under the Apache 2.0 License. Trained checkpoints,
when released, will inherit the licenses of their base models.
