# lgkit: แยก kernel ออกจาก lgtools + pattern `approval_gate`

- วันที่: 2026-09-26
- สถานะ: รอ review
- Repo: `langgraph-agent` (library `lgkit`) และ `laggraph-hermes/backend` (`lgtools`, ผู้ใช้หลักของ library)

## 1. เป้าหมาย

ทำให้การสร้าง agent workflow สะดวกขึ้นสำหรับเจ้าของโปรเจกต์เอง (ผู้ใช้คนเดียว) โดย:

1. ย้ายแกนสร้าง graph ของ `lgtools` (spec → `StateGraph`) มาเป็น library `lgkit` ที่ไม่มี UI/HTTP/DB — **ต้นฉบับเดียว**, `lgtools` import กลับมาใช้
2. เพิ่ม pattern สำเร็จรูปตัวแรก: `approval_gate` — จุดอนุมัติที่เลือกได้ว่าคน หรือ agent เป็นผู้ตัดสิน

### ไม่อยู่ในขอบเขตรอบนี้
- pattern อื่น (plan-execute-review, router, fan-out/committee) — ทำรอบถัดไป
- ย้าย `runner.py`, `session.py`, `harness/`, `*_store.py`, `worker.py`, `testing.py`/`TestCaseSpec` — ยังอยู่ที่ `lgtools`
- node kind อีก 11 ตัวของ `lgtools` (agent, react_agent, swarm, supervisor, query ฯลฯ) — อยู่ที่ `lgtools` และ register เข้า registry ของ `lgkit`
- Fluent builder / decorator / YAML API

## 2. ที่มา (self-audit 2026-09-26)

สำรวจโปรเจกต์ agent ของตัวเอง 8 ตัวใน `~/Project/product/`:

- `laggraph-hermes` (`lgtools`, 514 commits) มี kernel สมบูรณ์ที่สุด — `WorkflowSpec`/`NodeSpec`/`EdgeSpec` + `graph_builder.py` + node 13 kind
- LLM client factory ถูกเขียนใหม่ 4 ครั้งใน 4 โปรเจกต์
- ดึง JSON จาก LLM ด้วย regex + fallback เองใน 3 โปรเจกต์
- HITL พลาด 2 ครั้ง: `ai-agent-designer` เป็น stub คืน `approved: true` เสมอ; `lgtools` ต้องเตือนว่าโค้ดก่อน `interrupt()` รันซ้ำตอน resume
- โปรเจกต์ที่เริ่มใหม่จากศูนย์ถูกทิ้ง (`my-ai-hermes` 1 commit, `ai-agent-designer`) → จึงเลือกแยกของที่ใช้งานจริงแล้ว แทนการเขียนใหม่

## 3. พฤติกรรมของ `approval_gate`

### 3.1 โหมด

| ตั้งค่า | พฤติกรรม |
|---|---|
| `approver="human"` (ค่าเริ่มต้น) | `interrupt()` รอคนตอบ |
| `approver="human", advise=True` | node `advisor` ให้ LLM แนะนำ choice + เหตุผลก่อน แล้ว `interrupt()` รอคนยืนยัน/เปลี่ยน — agent ไม่ตัดสินเอง |
| `approver="agent"` | `advisor` ตัดสิน; ถ้าตอบ `escalate`, confidence < `min_confidence`, หรือ advisor ล้มเหลว → `interrupt()` ส่งต่อให้คน |

### 3.2 API

```python
from lgkit.patterns import approval_gate

frag = approval_gate(
    "publish_check",                       # id ของ gate node; advisor node = "publish_check__advisor"
    message="Approve this draft? {draft}", # format ด้วย state (flatten) เหมือน hitl เดิม
    choices=["approve", "revise"],
    approver="human",                      # "human" | "agent"
    advise=False,                          # ใช้ได้เฉพาะ approver="human"; approver="agent" มี advisor เสมอ
    criteria=None,                         # เกณฑ์ให้ advisor — บังคับเมื่อมี advisor
    min_confidence=0.7,                    # ใช้เฉพาะ approver="agent"
    llm=None,                              # {"provider": ..., "model": ...}; None = ค่าเริ่มต้นของ build_llm
    result_key=None,                       # None = ใช้ id
)
# frag: Fragment(nodes: list[NodeSpec], edges: list[EdgeSpec], entry: str, exit: str)
# entry = advisor id (ถ้ามี) ไม่งั้น gate id; exit = gate id
# ผู้ใช้ต่อ edge เข้า frag.entry และ route ออกจาก frag.exit ด้วย condition = choice
# frag.to_workflow(name=...) → WorkflowSpec ที่มี gate เดี่ยว ๆ (ใช้ใน test/ตัวอย่าง)
```

`approval_gate` คืน **ข้อมูล spec ล้วน** (ไม่มี runtime ของตัวเอง) → เปิด/แก้ใน visual editor ของ `lgtools` ได้

### 3.3 Node kinds ใหม่

- `advisor` — เรียก LLM ด้วย `with_structured_output` schema `{choice: Literal[*choices, "escalate"], confidence: float 0–1, reason: str}` เก็บผลใน `scratch[f"{id}__advice"]` เป็น `{"choice", "confidence", "reason"}` หรือ `{"error": str}` เมื่อล้มเหลว (หลัง retry ตาม `NodePolicy.retry`) **ไม่ raise**
- `approval` — อ่าน advice จาก scratch แล้ว:
  - `approver="agent"` และ advice สำเร็จ, `choice != "escalate"`, `confidence >= min_confidence` → ตัดสินโดย agent ไม่ interrupt
  - กรณีอื่น → `interrupt({"node", "message", "choices", "advice", "advice_error"})`
  - **ห้ามมี side effect ก่อนบรรทัด `interrupt()`** (LangGraph รัน node ใหม่ตั้งแต่ต้องตอน resume) — นี่คือเหตุผลที่ LLM อยู่ใน `advisor` แยกต่างหาก ผลถูก checkpoint ไว้แล้ว

### 3.4 ผลลัพธ์ (เหมือนกันทุกโหมด)

```python
{
  "signal": choice,
  "scratch": {result_key: {
      "choice": str, "comment": str,
      "by": "human" | "agent",
      "advice": {...} | None,
  }},
}
```

graph ส่วนที่เหลือ route ด้วย `signal` เหมือน hitl เดิม ไม่ต้องรู้ว่าใครตัดสิน

## 4. โครงสร้าง package

```
langgraph-agent/
├── pyproject.toml          # name="langgraph-agent", package lgkit, requires-python>=3.12
├── src/lgkit/
│   ├── __init__.py
│   ├── spec.py             # WorkflowSpec, NodeSpec, EdgeSpec, NodePolicy, RetryConfig, CacheConfig,
│   │                       # StateField, IOField, AgentSpec ฯลฯ — ไม่รวม *Request (HTTP)
│   ├── registry.py         # ย้ายตรง
│   ├── state.py            # ย้ายตรง (AgentState, reducers)
│   ├── context.py
│   ├── events.py
│   ├── hooks.py            # BuildHooks
│   ├── builder.py          # build_graph
│   ├── llm.py              # build_llm(provider, model, **kw) — ไม่อ่าน lgtools Settings
│   ├── nodes/{hitl,advisor,approval}.py
│   └── patterns/approval_gate.py
├── tests/
└── examples/
    ├── plan_execute_review.py   # main.py เดิม ย้ายมา + แก้บั๊ก (ข้อ 7)
    └── approval_demo.py
```

### 4.1 การกลับทิศ dependency

สิ่งที่ kernel เดิม import จาก `lgtools` จะกลายเป็นค่าที่ส่งเข้ามา:

| เดิม | ใน lgkit |
|---|---|
| `lgtools.api_models.StrictRequest` | ไม่ใช้ใน `spec.py`; `*Request` อยู่ `lgtools` |
| `lgtools.config.SUPPORTED_PROVIDERS`, `lgtools.llm.DEFAULT_MODELS` (validator ของ `AgentSpec`) | ค่าคงที่ใน `lgkit.llm`; `lgtools.config` import จาก `lgkit` |
| `lgtools.config.Settings` ใน `llm.py` | `build_llm` รับ key/base_url เป็นพารามิเตอร์; `lgtools.llm` เป็น wrapper ที่อ่าน Settings แล้วเรียก `lgkit.llm.build_llm` |
| `agents.resolve_agent`, `workflows.resolve_workflow`, `resolved.Resolver`, `event_dispatch.dispatch_event` ใน `graph_builder` | ฟิลด์ของ `BuildHooks` |

```python
@dataclass(frozen=True)
class BuildHooks:
    agent_resolver: Callable[[str], AgentSpec | None] = lambda _id: None
    workflow_resolver: Callable[[str], WorkflowSpec | None] = lambda _id: None
    resolve: Callable[[WorkflowSpec], Any] = lambda spec: None   # → ค่า `resolved` ที่ส่งให้ node fn
    dispatch_event: Callable[..., Any] | None = None             # None = ไม่ dispatch event script

def build_graph(spec, checkpointer=None, store=None, dry_run=False, *, hooks=DEFAULT_HOOKS, resolved=None): ...
```

ไม่มี state แบบ global (`configure()`) — hooks ส่งต่อการเรียกแต่ละครั้งเท่านั้น

### 4.2 ฝั่ง `lgtools`

- `uv add --editable ../../langgraph-agent`
- ไฟล์เดิมกลายเป็น shim เพื่อไม่ต้องแก้ importer 72 ไฟล์:
  - `kernel/schemas.py` → `from lgkit.spec import *` + คง `*Request` ไว้
  - `kernel/registry.py`, `kernel/state.py`, `kernel/context.py`, `kernel/events.py` → re-export
  - `kernel/graph_builder.py` → `build_graph` ที่ unwrap `ResolvedSpec` แล้วเรียก `lgkit.build_graph(..., hooks=LGTOOLS_HOOKS, resolved=...)`; re-export `GraphBuildError` และ helper ที่ test เดิมใช้
  - `templates/primitives/hitl_node.py` → ใช้ `lgkit.nodes.hitl`
- node kind ของ `lgtools` register เข้า registry ของ `lgkit` ตอน import เหมือนเดิม

### 4.3 ลำดับการย้าย (ขั้นถัดไปเริ่มเมื่อ test ขั้นก่อนผ่าน)

0. บันทึก baseline: รัน test ทั้งหมดของ `lgtools` เก็บผล (pass/fail/skip ต่อไฟล์)
1. สร้าง `lgkit` — ก๊อปโมดูล + test ที่เกี่ยวข้อง (`test_graph_builder`, `test_kernel_state`, `test_cycle_detection`, `test_iteration_guard`, `test_context` ฯลฯ เท่าที่ไม่ต้องพึ่ง runner) ปรับ import ให้ผ่านใน `lgkit`
2. เปลี่ยน `lgtools` เป็น shim → ผล test ต้องตรง baseline ทุกไฟล์
3. เพิ่ม `advisor`, `approval`, `approval_gate` ใน `lgkit` (TDD)
4. ย้าย `main.py` → `examples/` + แก้บั๊ก, เพิ่ม `approval_demo.py`

## 5. Error handling

หลัก: **เมื่อไม่แน่ใจ ส่งให้คนตัดสิน ไม่ approve อัตโนมัติ**

| สถานการณ์ | พฤติกรรม |
|---|---|
| advisor เรียก LLM ล้มเหลวหลัง retry | บันทึก `{"error": ...}`; โหมด human → interrupt พร้อม `advice=None`, `advice_error`; โหมด agent → interrupt (escalate) |
| structured output parse ไม่ผ่าน / choice ไม่อยู่ใน choices | ถือเป็นความล้มเหลวของ advisor (แถวบน) |
| confidence < `min_confidence` หรือ `escalate` | interrupt |
| คน resume ด้วย choice ที่ไม่อยู่ใน `choices` | `ValueError` |
| spec มี `approval` node ที่อาจ interrupt แต่ `checkpointer=None` | `GraphBuildError` ตอน `build_graph` พร้อมข้อความแนะนำ `InMemorySaver`/`SqliteSaver` |
| `approver="agent"` หรือ `advise=True` แต่ไม่มี `criteria`; `advise=True` คู่กับ `approver="agent"`; `choices` ว่างหรือมี `"escalate"` | `ValueError` ตอนเรียก `approval_gate()` |

หมายเหตุ: gate โหมด `agent` ยังต้องมี checkpointer เพราะอาจ escalate

## 6. การทดสอบ

- LLM ปลอม: `langchain_core` `GenericFakeChatModel` (ไม่เรียก API จริงใน test)
- Unit ต่อทุกแถวของตาราง error และทุกโหมดในข้อ 3.1
- Human flow: `InMemorySaver` → invoke จนถึง interrupt → ตรวจ payload → `Command(resume={"choice": ..., "comment": ...})` → ตรวจ state ปลายทางและ `signal`
- **Regression ของปัญหาเดิม:** นับการเรียก LLM ของ advisor ต้องเท่ากับ 1 แม้ผ่าน interrupt + resume
- Migration: ผล test ของ `lgtools` หลัง shim ต้องตรง baseline ขั้น 0
- `examples/approval_demo.py` รันได้ด้วย LLM ปลอมโดยไม่ต้องมี API key; สลับ LLM จริงด้วย env

## 7. งานเล็กที่ทำไปพร้อมกัน (จาก review `main.py`)

- reviewer `len(result) > 10` เป็นจริงเสมอ และ executor คืนค่าเดิมทุกรอบ → วนไม่จบถ้าไม่ผ่าน: เพิ่ม `attempts` + `max_rounds` ใน `route`
- ใส่ type ให้ node (`state: AgentState`), ย้าย `invoke` ไปใต้ `if __name__ == "__main__":`
- `pyproject.toml`: ใส่ description, ตัด `openai` (ไม่ใช้ตรง ๆ; provider SDK มาผ่าน `langchain-*`), ลด `requires-python` เป็น `>=3.12`
- `.gitignore`: เพิ่ม `.env`, `.DS_Store`; README สั้น ๆ

## 8. ความเสี่ยง / สิ่งที่ต้องยืนยันตอนทำแผน

- `graph_builder.py` ใช้ `ResolvedSpec`/`Resolver` ลึกแค่ไหน — ถ้า node fn ของ lgkit เองต้องใช้ `resolved` มากกว่า `None` ได้ ต้องออกแบบ `hooks.resolve` ให้พอ
- `schemas.py` 1,037 บรรทัด อาจมีส่วนที่อ้าง store/harness เพิ่มเติมจากที่สำรวจ — แยกทีละ class
- `lgtools` และ `langgraph-agent` ต้องใช้ `langgraph`/`langchain-core` เวอร์ชันที่เข้ากันได้ (ปัจจุบัน `lgtools` pin `langchain==1.3.*`)
