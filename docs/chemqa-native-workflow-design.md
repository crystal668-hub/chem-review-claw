# ChemQA Native Workflow 改动设计方案

日期：2026-04-11  
状态：待审核  
适用范围：
- DebateClaw V1：multi-agent runtime substrate
- chemqa-review：domain-specific workflow package

---

## 1. 背景与问题

当前 `chemqa-review` 运行在 DebateClaw V1 的 `review-loop@1` 之上，并通过 bridge/runtime workaround 兼容 ChemQA 的单主候选 + 固定 reviewer 语义。

现状冲突：

- `review-loop@1` 底层假设 5 个 proposer 互相 review，一轮 review 期望总数为 20。
- ChemQA 真实语义是：
  - 只有 `proposer-1` 是 candidate owner
  - `proposer-2..5` 是固定 reviewer lanes
  - 每轮只需要 4 个 reviewer 对 `proposer-1` 的 submission 做 formal review
- 当前 bridge 方案虽然能跑通，但存在：
  - 16 条语义无意义的 transport review
  - runtime/collector 中存在大量“忽略非 candidate review”的兼容逻辑
  - token/步骤成本偏高
  - engine 语义与 domain 语义长期分裂，维护风险高

目标是把 `chemqa-review` 改造成真正的 native workflow，同时保留 DebateClaw V1 作为通用 multi-agent runtime substrate。

---

## 2. 定位与边界

### 2.1 DebateClaw V1 的定位

DebateClaw V1 不再被视为 ChemQA 的工作流语义引擎，而是通用运行底座（runtime substrate）。

**负责：**

- fixed debate slots / agent runtime
- OpenClaw wrapper 与 slot workspace 生命周期
- session 隔离与 session-id 绑定
- run launch / team bootstrap / materialization
- command map / prompt bundle 等运行装配
- 通用 run state host 与 workflow package 加载
- 通用命令面：`status` / `next-action` / `submit-*` / `advance` / `summary`

**不负责：**

- ChemQA 的 acceptance 语义
- proposer-main / reviewer lane 的 domain 含义
- “一轮 review 应该是 4 条还是 20 条”这类 domain policy
- ChemQA 最终 protocol / artifact reconstruction 规则

### 2.2 chemqa-review 的定位

chemqa-review 是 domain-specific workflow package。

**负责：**

- ChemQA phase graph
- role semantics
- next-action policy
- review target graph
- submission/review/rebuttal 的 domain 校验
- acceptance / rejection / failure 判定
- protocol scaffold 与最终 artifact reconstruction 语义
- coordinator 终态模型整理逻辑

---

## 3. 总体方案

### 3.1 方案摘要

保留外部入口 `chemqa-review@1`，但其底层不再绑定 DebateClaw V1 的 `review-loop@1`。

改为：

- DebateClaw V1 提供“可加载 workflow package”的运行宿主
- chemqa-review 提供自己的 `ChemQAWorkflow` 实现
- 运行时由 chemqa-review launcher/materialize 将 `ChemQAWorkflow` 挂载到 DebateClaw V1 substrate 上

### 3.2 核心原则

1. **不再要求 all-to-all review**
2. **只保留 proposer-main 与四个固定 reviewer 的真实语义**
3. **保留现有 fixed slots / OpenClaw runtime / model slot wiring / launch pipeline 的复用价值**
4. **接口清晰：workflow package 决定“怎么推进”，substrate 决定“怎么运行”**

---

## 4. Native ChemQA Workflow 设计

### 4.1 角色定义

- `debate-coordinator`: 协调、终态决策、最终协议整理
- `proposer-1`: 唯一 candidate owner
- `proposer-2`: search_coverage reviewer
- `proposer-3`: evidence_trace reviewer
- `proposer-4`: reasoning_consistency reviewer
- `proposer-5`: counterevidence reviewer

### 4.2 状态机

建议 phase 设计：

- `propose`
- `review`
- `rebuttal`
- `done`
- `failed`

### 4.3 Phase 行为

#### propose

- `proposer-1`：`propose`
- `proposer-2..5`：`wait`
- `debate-coordinator`：`wait`

推进条件：
- `proposer-1` 提交有效 candidate submission

#### review

- `proposer-2..5`：`review`
- review target 固定为 `proposer-1`
- `proposer-1`：`wait`
- `debate-coordinator`：`wait/advance`

推进条件：
- 4 个固定 reviewer lane 全部对当前 candidate revision 提交 formal review

#### rebuttal

- `proposer-1`：`rebuttal`
- `proposer-2..5`：`wait`
- `debate-coordinator`：`wait`

推进条件：
- proposer-main 提交有效 rebuttal / candidate update

#### done

- coordinator 使用模型整理最终 protocol / final answer
- collector 生成最终 artifacts

#### failed

进入条件：
- recovery/repair budget exhausted
- required reviewer lane 无法完成
- artifact 连续不合法且超出预算
- runtime liveness failure 无法恢复

### 4.4 Review 数量

Native workflow 下，一轮 review 的完成条件明确为：

- 只要求 4 条 formal review：
  - `proposer-2 -> proposer-1`
  - `proposer-3 -> proposer-1`
  - `proposer-4 -> proposer-1`
  - `proposer-5 -> proposer-1`

不再有：
- reviewer 之间互评
- proposer-main 去 review reviewer lane
- transport placeholder review

---

## 5. 接口设计

以下接口由 DebateClaw V1 substrate 提供，workflow package 必须实现或适配。

### 5.1 Workflow Package 接口

建议在 `chemqa-review/runtime/workflow.py` 中实现一个 `ChemQAWorkflow` 类，包含：

```python
class ChemQAWorkflow:
    workflow_id: str
    version: str
    roles: list[str]

    def initialize_run(self, run_config: dict) -> dict: ...
    def compute_next_action(self, state: dict, role: str) -> dict: ...
    def submit_artifact(self, state: dict, role: str, artifact_type: str, payload: dict) -> dict: ...
    def advance(self, state: dict) -> dict: ...
    def build_status(self, state: dict, role: str) -> dict: ...
    def build_summary(self, state: dict) -> dict: ...
    def finalize(self, state: dict) -> dict: ...
```

### 5.2 接口语义

#### `initialize_run(run_config)`
初始化 workflow state。

输出至少包含：
- `phase`
- `candidate_owner`
- `required_reviewer_lanes`
- `review_round`
- `rebuttal_round`
- `candidate_revision`
- round budgets

#### `compute_next_action(state, role)`
返回当前 role 的动作描述。

可能动作：
- `propose`
- `review`
- `rebuttal`
- `wait`
- `advance`
- `stop`

review 动作必须包含：
- `target_owner: proposer-1`
- `candidate_revision`
- `review_round`

#### `submit_artifact(state, role, artifact_type, payload)`
接收 artifact 提交并更新 state。

artifact types 至少包括：
- `candidate_submission`
- `formal_review`
- `rebuttal`
- `coordinator_protocol`
- `terminal_failure`

#### `advance(state)`
按 ChemQA 语义推进 phase。

#### `build_status(state, role)`
输出 role-aware 状态。

#### `build_summary(state)`
输出全局 summary，供 finalization / collector / coordinator 使用。

#### `finalize(state)`
输出 deterministic final scaffold，供 coordinator 终态模型整理与 collector 使用。

---

## 6. 能力边界

### 6.1 DebateClaw V1 substrate 提供的能力

必须保留：

1. **slot runtime 能力**
   - 固定 slot agent 调起
   - slot workspace 隔离
   - session-id 绑定

2. **运行宿主能力**
   - team lifecycle
   - role process spawning
   - workflow package 动态加载

3. **通用状态宿主能力**
   - 读取/写入 workflow state
   - 基础 event log / artifact registry

4. **通用命令面能力**
   - `status`
   - `next-action`
   - `submit-*`
   - `advance`
   - `summary`

5. **装配能力**
   - model slot wiring
   - prompt bundle materialization
   - command map materialization
   - run launch

### 6.2 chemqa-review package 提供的能力

必须接管：

1. **ChemQA phase logic**
2. **role semantics**
3. **review graph**
4. **artifact schema / validators**
5. **acceptance/rejection/failure semantics**
6. **protocol scaffold**
7. **coordinator final model turn 内容约束**
8. **collector-compatible summary surface**

### 6.3 明确不做的事情

chemqa-review native workflow **不再**：
- 模拟 `review-loop@1` 的 20 review transport
- 引入 placeholder review 作为 engine 兼容层
- 维护 reviewer-reviewer transport review
- 以“忽略 16 条噪声 review”的方式达成 acceptance

---

## 7. 状态数据结构建议

### 7.1 Workflow State

```json
{
  "workflow_id": "chemqa-review@1",
  "phase": "review",
  "status": "running",
  "candidate_owner": "proposer-1",
  "candidate_revision": 1,
  "review_round": 1,
  "rebuttal_round": 0,
  "max_review_rounds": 3,
  "max_rebuttal_rounds": 2,
  "required_reviewer_lanes": [
    "proposer-2",
    "proposer-3",
    "proposer-4",
    "proposer-5"
  ],
  "candidate_submission": {},
  "reviews_by_round": {},
  "rebuttals": [],
  "acceptance_status": null,
  "failure_reason": null
}
```

### 7.2 Status 输出

```json
{
  "workflow": "chemqa-review@1",
  "phase": "review",
  "status": "running",
  "candidate_owner": "proposer-1",
  "candidate_revision": 1,
  "review_round": 1,
  "rebuttal_round": 0,
  "review_completion": {
    "expected": 4,
    "submitted": 3,
    "missing_lanes": ["proposer-4"]
  },
  "blocking_lanes": ["proposer-3"]
}
```

---

## 8. coordinator 终态行为

### 8.1 新行为

coordinator 在 terminal 阶段需要：

1. 读取 workflow `summary`
2. 使用 deterministic `finalize(state)` scaffold 作为输入
3. 启动一次真实模型 turn
4. 生成 `chemqa_review_protocol.yaml`
5. runtime 做 protocol schema 校验
6. collector 生成最终 artifacts

### 8.2 设计要求

- coordinator 必须在终态触发 session，保留可审计轨迹
- deterministic scaffold 仍然保留，作为兜底
- 模型输出失败时可 fallback 到 deterministic scaffold，避免 run 无法完成

---

## 9. 改动点清单

以下按“substrate / package”拆分。

### 9.1 DebateClaw V1 侧改动（最小必要）

> 注意：这里的目标不是新增 ChemQA 语义，而是让 DebateClaw V1 具备“加载 workflow package”的能力。

#### 新增/改造能力

1. **Workflow package loader**
   - 能从 runplan/runtime context 中读取 workflow package 描述
   - 支持通过 Python path/module 动态加载 workflow implementation

2. **Generic state host**
   - 将当前 `review-loop` 特定状态机逻辑外提为 package hook 调用

3. **Generic command bridge**
   - `status`
   - `next-action`
   - `submit-*`
   - `advance`
   - `summary`
   这些命令改为转发给 workflow package

#### 可能涉及文件（设计级，不是最终唯一清单）

- `~/.openclaw/skills/debateclaw-v1/scripts/debate_state.py`
- `~/.openclaw/skills/debateclaw-v1/scripts/prepare_debate.py`
- `~/.openclaw/skills/debateclaw-v1/scripts/debate_templates.py`
- 新增如：
  - `.../scripts/workflow_host.py`
  - `.../scripts/workflow_loader.py`
  - `.../scripts/workflow_api.py`

### 9.2 chemqa-review 侧改动（主体）

#### 新增模块

建议新增目录：

```text
~/.openclaw/skills/chemqa-review/runtime/
  workflow.py
  state_models.py
  next_action.py
  transitions.py
  submissions.py
  summary.py
  finalization.py
```

#### 现有脚本改造

1. `compile_runplan.py`
   - 不再引用 `review-loop@1` 作为 engine workflow
   - 改为在 runplan 中写入 workflow package 描述

2. `materialize_runplan.py`
   - 将 workflow package 信息写入 runtime context
   - 继续负责 slot model apply / command map / prompt bundle

3. `launch_from_preset.py`
   - 保持外部入口不变，内部改走 package workflow 模式

4. `chemqa_review_openclaw_driver.py`
   - 删除 `ensure_transport_review()` 桥接语义
   - review 阶段只处理 reviewer -> proposer-1 formal review
   - propose/rebuttal 只处理 proposer-main
   - coordinator 终态继续保留模型整理 protocol

5. `chemqa_review_artifacts.py`
   - 简化掉大量 transport review 忽略/兼容逻辑
   - 直接围绕 1 candidate + 4 reviewer + N rebuttal cycles 重建

6. prompts
   - 删除/废弃 `review-loop-bridge.md`
   - 新增 `chemqa-native-workflow.md`

#### 可能涉及文件

- `~/.openclaw/skills/chemqa-review/scripts/compile_runplan.py`
- `~/.openclaw/skills/chemqa-review/scripts/materialize_runplan.py`
- `~/.openclaw/skills/chemqa-review/scripts/launch_from_preset.py`
- `~/.openclaw/skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`
- `~/.openclaw/skills/chemqa-review/scripts/chemqa_review_artifacts.py`
- `~/.openclaw/skills/chemqa-review/prompts/modules/policies/review-loop-bridge.md`（废弃）
- 新增 `~/.openclaw/skills/chemqa-review/prompts/modules/policies/chemqa-native-workflow.md`

---

## 10. MVP（最小可实施版本）

建议按以下阶段执行。

### Phase 0：确认设计边界（本次审核）
需要你确认：
- `chemqa-review@1` 外部命名保持不变
- DebateClaw V1 只做 substrate，不再内置 ChemQA 语义
- coordinator 终态必须有模型 turn

### Phase 1：substrate 最小 host 能力
目标：让 DebateClaw V1 可以装载 workflow package。

交付：
- workflow loader
- generic state host skeleton
- generic `status/next-action/advance/summary` hook 分发

### Phase 2：chemqa-review native workflow package
目标：实现 `ChemQAWorkflow`。

交付：
- propose/review/rebuttal/done/failed 状态机
- 4-review completion 规则
- role-aware next-action

### Phase 3：driver / prompt / finalization 切换
目标：移除 review-loop bridge 依赖。

交付：
- driver 精简
- prompt policy 更新
- finalization 接口对齐

### Phase 4：collector 对齐与回归
目标：保证 artifacts 输出接口兼容。

交付：
- accepted / rejected / failure / rebuttal 回归样例
- coordinator session 存在验证
- reviewer 实际模型映射验证

---

## 11. 验收标准

### 11.1 运行层

- fresh run 正常启动
- fixed slots 使用正确模型：
  - coordinator = su8/gpt-5.4
  - proposer-main = su8/gpt-5.4
  - reviewers = packy/gpt-5.4
- coordinator 在 terminal 阶段产生 session

### 11.2 语义层

- `propose` 只要求 1 个 candidate submission
- `review` 每轮只要求 4 条 reviewer -> proposer-main formal review
- status 中 review expected = 4，而不是 20
- 不再生成 reviewer 互评 transport review

### 11.3 artifact 层

- `chemqa_review_protocol.yaml` 由 coordinator 模型终态生成
- collector 输出接口保持兼容
- accepted/rejected/failure 判定符合 ChemQA 规则

---

## 12. 风险与注意事项

### 风险 1：substrate 抽象不彻底
如果 DebateClaw V1 的通用接口抽象不够干净，后面容易重新把 ChemQA 语义塞回 substrate。

### 风险 2：chemqa-review package 过重
如果 package 里塞入过多 runtime 细节，可能导致事实上的“小型 engine fork”。

### 风险 3：collector 兼容性
需要确保新的 summary/state 能继续支持现有 artifact surface。

### 风险 4：迁移期间双轨维护
review-loop bridge 与 native workflow 需要短期并存，避免直接切换造成回归不可控。

---

## 13. 审核结论需要确认的点

请重点确认以下事项：

1. **入口命名**
   - 对外保持 `chemqa-review@1`
   - 内部改为 workflow package 模式

2. **DebateClaw V1 定位**
   - 仅保留为 multi-agent runtime substrate
   - 不再承担 ChemQA workflow 语义

3. **Native review 数量**
   - 每轮 review 只要求 4 条 reviewer -> proposer-main formal review

4. **Coordinator 终态行为**
   - 必须用模型整理并生成最终 protocol / answer

5. **实施顺序**
   - 先做 substrate hook，再做 chemqa-review workflow package，再切换 launcher

---

## 14. 建议的执行入口（审核通过后）

审核通过后，建议直接按以下顺序实施：

1. DebateClaw V1：新增 workflow host / loader 骨架
2. chemqa-review：新增 `runtime/` workflow package
3. chemqa-review：切 compile/materialize/launch 到 package workflow
4. chemqa-review：简化 driver 与 artifact reconstruction
5. 跑一轮 fresh ChemQA run 验证模型、review 数量、coordinator session、最终 artifacts

---

## 15. 当前建议

**建议批准此方案，并按 MVP 顺序实施。**

此方案能最大程度保持两者边界清晰：
- DebateClaw V1 继续作为通用运行底座
- chemqa-review 真正成为 ChemQA 协议包

同时避免继续在 `review-loop@1` 上追加 bridge/compat 逻辑。
