# ADR-0002：用未来公开信标生成 first-reveal cases

## Status

Accepted。

## Context

固定本地 seed 虽然容易复现，但实验准备者能在冻结前查看对应场景。项目需要证明实现、策略和
门槛先冻结，最终评测数据随后才变得可知。

## Decision

在冻结协议中预先指定一个尚未发布的 drand Quicknet round。信标发布后，先用两路 relay 和
固定客户端验证，再由 protocol hash 与 beacon randomness 派生 blind root、scenario seed
和 noise seed。

## Consequences

数据派生过程公开且可复核，也多了一项 Node/drand 依赖。first reveal 只能运行一次；结果被
查看后，这批 cases 立刻转为公开验证集，后续调参不能继续把它们称为 blind evidence。

## Evidence

冻结条件见 [protocol](../../results/franka_safety_preholdout/protocol.json)，首次揭盲证据见
[reveal](../../results/franka_safety_blind/reveal.json)，验证器见
[`verify_drand_beacon.mjs`](../../tools/verify_drand_beacon.mjs)。
