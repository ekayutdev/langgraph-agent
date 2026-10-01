# lgkit: node `llm`, node `loop_limit` และ pattern `plan_execute_review`

- วันที่: 2026-10-01
- สถานะ: รอ review
- Repo: `langgraph-agent` (library `lgkit`) และ `laggraph-hermes` (`lgtools` — ต้องรองรับ node ชนิดใหม่ใน editor)
- ต่อจาก: `2026-09-26-lgkit-approval-gate-design.md`

## 1. เป้าหมาย

เพิ่ม pattern สำเร็จรูปตัวที่สอง: วางแผน → ลงมือทำ → ตรวจ → วนแก้จนผ่านหรือครบจำนวนรอบ โดย

1. เพิ่ม node ชนิดทั่วไป `llm` (เรียก LLM หนึ่งครั้ง) ที่ pattern ถัดไป (router, committee) ใช้ต่อได้
2. เพิ่ม node `loop_limit` (นับรอบ ไม่ใช้ LLM)
3. เพิ่ม `plan_execute_review()` ที่คืน `Fragment` (ข้อมูล spec ล้วน เหมือน `approval_gate`)

### สิ่งที่ผู้ใช้ตัดสินใจแล้ว
- ขั้น execute: ค่าเริ่มต้นเป็น LLM เขียนข้อความ และเปลี่ยนเป็น node ของผู้ใช้เองได้
- reviewer ไม่ผ่าน → วนกลับ executor พร้อม feedback (แผนเดิมยังใช้อยู่)
- ใช้ node `llm` ตัวเดียวแบบทั่วไป ไม่สร้าง node เฉพาะทาง 3 ชนิด

### ไม่อยู่ในขอบเขตรอบนี้
- agent ที่มี tool / tool loop
- pattern router, committee
- ปุ่มใน editor ที่วาง pattern ทั้งชุดในคลิกเดียว
- การ push `lgtools`

## 2. หลักที่ต้องรักษา

- **ครบจำนวนรอบแล้วยังไม่ผ่าน ไม่ถือว่าอนุมัติ** (หลักเดียวกับ approval gate: ไม่แน่ใจ ไม่ approve เอง)
- **signal ที่ไม่ตรงเงื่อนไขใดต้องไม่ไหลไป edge แรก** — จบ run แทน
- **dry-run ไม่สร้าง LLM จริงไม่ว่ากรณีใด**
- `lgkit` ไม่ import `lgtools`; ไม่มี config แบบ global

## 3. Node `llm`

เรียก LLM หนึ่งครั้ง เก็บผลใน `scratch` และ (ถ้าตั้งไว้) ส่ง `signal`

### 3.1 พารามิเตอร์

| ชื่อ | ความหมาย | ค่าเริ่มต้น |
|---|---|---|
| prompt ของ node (`NodeSpec.prompt`) | system prompt | `""` (`uses_node_prompt=True`) |
| `message` | template ของข้อความ human; render ด้วย `flat_state` (`{task}`, `{scratch.x.y}`; key ที่ไม่มี = ค่าว่าง) | `"{task}"` |
| `result_key` | key ใน `scratch` ที่เก็บผล | id ของ node |
| `output_fields` | `[{name, type, description?, required?}]` — `type` ∈ string, number, boolean, object, array; ถ้าใส่ ผลเป็น dict | ไม่ใส่ → ผลเป็น `str` |
| `signal_field` | ชื่อ field ใน `output_fields` ที่ใช้เป็น `signal` | ไม่มี |
| `signal_values` | ค่าที่อนุญาตของ `signal_field` (บังคับด้วย `Literal`) — ต้องใส่เมื่อมี `signal_field` | ไม่มี |
| `llm` | `{provider, model, ...}` เฉพาะ node นี้ | ใช้ LLM ของ run |
| `max_attempts` | จำนวนครั้งที่ลอง (เรียกไม่สำเร็จ หรือผลไม่ตรงรูปแบบ) | 2 |

### 3.2 พฤติกรรม

- มี `output_fields` → `llm.with_structured_output(<pydantic model ที่สร้างจาก fields>)`; ไม่ดึง JSON ด้วย regex. ผลที่เป็น `BaseMessage` (จาก fake model ใน dry-run) จะ parse `.content` เป็น JSON แล้ว validate
- ไม่มี `output_fields` → `llm.invoke(...)` แล้วเก็บ `.content` เป็น `str`
- คืนค่า `{"scratch": {result_key: <ผล>}, "events": [{"node": id, "result": <ผล>, "tools": []}]}` และ `"signal": <ค่า>` เมื่อมี `signal_field`
- **ล้มเหลวหลังลองครบ → raise** `RuntimeError("llm node '<id>' failed after N attempts: <สาเหตุล่าสุด>")` (ต่างจาก `advisor` ที่บันทึก error แล้วให้คนตัดสิน เพราะ node นี้ไม่มีคนรอรับ)
- dry-run: ถ้า `ctx.dry_run` มี `next_mock(node_id)` (เช่น `DryRunState` ของ `lgtools`) node จะดึงคำตอบถัดไปจากตรงนั้นทุกครั้งที่ถูกรัน — `model_for()` สร้าง model ปลอมใหม่ทุกครั้ง จึงตอบคำตอบแรกซ้ำเมื่อ node อยู่ใน loop; ถ้า script หมด (`None`) ถือเป็นความล้มเหลวหนึ่งครั้ง
- ไม่มี tool, ไม่มี loop ภายใน

### 3.3 การเลือก LLM (ใช้ร่วมกับ `advisor`)

แยกเป็น `lgkit/nodes/_llm.py::resolve_llm(params, ctx, node_id)` แล้วให้ `advisor` เปลี่ยนมาใช้ (พฤติกรรม `advisor` ต้องเท่าเดิม):

1. `ctx.dry_run` มีค่า → `ctx.dry_run.model_for(node_id)`
2. context ที่ผูกไว้ (`current_ctx()`) มี dry_run แต่ `ctx` ไม่มี → `RuntimeError("dry-run context lost: refusing to build a real model")`
3. `params["llm"]` → `build_llm(**params["llm"])`
4. `ctx.llm`
5. `build_llm()`

## 4. Node `loop_limit`

นับจำนวนครั้งที่ผ่าน node นี้ ไม่ใช้ LLM

| พารามิเตอร์ | ความหมาย |
|---|---|
| `max_rounds` | จำนวนรอบสูงสุด (จำนวนเต็ม ≥ 1) — บังคับ |
| `counter_key` | key ใน `scratch` ที่เก็บตัวนับ — ค่าเริ่มต้น `<id>__count` |
| `exhausted_key` | key ที่ตั้งเป็น `True` เมื่อครบ — ค่าเริ่มต้น `<id>__exhausted` |

พฤติกรรม: `count = scratch[counter_key] + 1`; ถ้า `count < max_rounds` → `signal="again"`; ไม่เช่นนั้น → `signal="exhausted"` และ `scratch[exhausted_key]=True`. เขียน `scratch[counter_key]=count` ทุกครั้ง

ความหมายของ `max_rounds`: จำนวนครั้งที่ขั้น execute ทำงานได้มากที่สุด (`max_rounds=3` → execute สูงสุด 3 ครั้ง, revise ที่วนกลับได้ 2 ครั้ง)

## 5. Pattern `plan_execute_review`

### 5.1 API

```python
from lgkit.patterns import plan_execute_review

per = plan_execute_review(
    "doc",                                    # prefix ของ node id และ scratch key
    planner="Break the task into steps.",     # system prompt ของ planner
    executor="Write the document following the plan.",  # str (prompt) หรือ NodeSpec
    reviewer="Approve only if every step of the plan is covered.",
    max_rounds=3,
    llm=None,                                 # {provider, model} ใช้กับ node llm ทั้งหมดของ pattern
    executor_result_key=None,                 # บังคับเมื่อ executor เป็น NodeSpec
)
```

### 5.2 โครง graph

```
<id>__plan ─► <id>__execute ─► <id>__review ──approve──► exit "approved"
                   ▲                │
                   │              revise
                   │                ▼
                   └──again── <id>__limit ──exhausted──► exit "exhausted"
```

| node | kind | เขียน scratch | หมายเหตุ |
|---|---|---|---|
| `<id>__plan` | `llm` | `<id>__plan` (str) | message: `{task}` |
| `<id>__execute` | `llm` หรือของผู้ใช้ | `<id>__result` (หรือ `executor_result_key`) | message รวม task, แผน, ผลงานรอบก่อน, feedback (รอบแรกสองค่าหลังว่าง) |
| `<id>__review` | `llm` (มีโครงสร้าง) | `<id>__review` = `{verdict, feedback}` | `signal_field="verdict"`, `signal_values=["approve","revise"]`; message รวม task, แผน, ผลงาน |
| `<id>__limit` | `loop_limit` | `<id>__round`, `<id>__exhausted` | `max_rounds` |

เมื่อ executor เป็น `NodeSpec`: ใช้ตามที่ให้มา (ไม่เปลี่ยน id), reviewer อ่านผลจาก `scratch[executor_result_key]`

### 5.3 `Fragment.exits`

เพิ่ม field `exits: dict[str, tuple[str, str]]` = ชื่อทางออก → `(node id, condition)`:

- `plan_execute_review`: `{"approved": ("<id>__review", "approve"), "exhausted": ("<id>__limit", "exhausted")}`; `entry` = `<id>__plan`; `exit` = `<id>__review`, `choices` = `()`
- `approval_gate`: `{choice: (gate id, choice) for choice in choices}` — field `exit`/`choices` เดิมยังอยู่ ใช้ได้เหมือนเดิม
- `to_workflow()` ต่อทุกทางออกใน `exits` ไป `END`
- ผู้ใช้ต่อ pattern กันด้วย `EdgeSpec(source=node, target=..., condition=cond)` จาก `exits[name]` (เช่น ต่อ "exhausted" เข้า `approval_gate(...).entry`)

## 6. การเปลี่ยนแปลงใน builder

`_signal_router(..., end_on_miss=True)` ปัจจุบันใช้เฉพาะต้นทางชนิด `approval`. ขยายเป็นชุดคงที่ `_END_ON_MISS_KINDS = {"approval", "llm", "loop_limit"}`: signal ที่ไม่ตรงเงื่อนไขใดของต้นทางชนิดเหล่านี้ → `END`

เหตุผล: `<id>__limit` เป็นต้นทางของ back-edge จึงได้ iteration guard ของ run; เมื่องบรอบรวมหมด guard ตั้ง `signal="max_iterations"` ซึ่งไม่ตรง `again`/`exhausted` — ถ้าไหลไป edge แรก (`again`) จะวนไม่จบ. node ชนิดอื่น (รวม `hitl`) คงพฤติกรรมเดิม

## 7. Error handling

| สถานการณ์ | พฤติกรรม |
|---|---|
| `llm` เรียกไม่สำเร็จ/ผลไม่ตรงรูปแบบ หลังลองครบ | raise → run ล้ม พร้อมชื่อ node |
| ครบ `max_rounds` แล้วยังไม่ผ่าน | ทางออก "exhausted", `scratch["<id>__exhausted"]=True` — ไม่ถือว่าอนุมัติ |
| งบรอบรวม (`max_iterations`) หมดก่อน | จบ run (ข้อ 6) |
| dry-run ไม่ได้เขียน mock ให้ node `llm` แบบมีโครงสร้าง | validate ไม่ผ่าน → run ล้ม; ข้อความ error บอกให้เพิ่ม `mock_scripts["<node id>"]` |
| ตั้งค่าผิดตอนเรียก `plan_execute_review()` | `ValueError`: `max_rounds < 1`; prompt ว่าง; executor เป็น `NodeSpec` แต่ไม่มี `executor_result_key`; executor เป็น str แต่ใส่ `executor_result_key`; id ของ executor ซ้ำกับ node ของ pattern |
| `llm`/`loop_limit` ที่เขียนสเปกเองแล้วพารามิเตอร์ผิด | fatal ตอน `build_graph` ผ่าน `node_validation`: `llm` — `signal_field` ไม่อยู่ใน `output_fields`, มี `signal_field` แต่ `signal_values` ว่าง; `loop_limit` — `max_rounds` ขาดหรือ < 1 |

## 8. การทดสอบ (TDD, ใช้ `ScriptedLLM`)

`ScriptedLLM` ต้องรองรับ `invoke()` ตรง ๆ (ข้อความธรรมดา) เพิ่มจาก `with_structured_output().invoke()`

1. `llm`: ข้อความธรรมดา; มีโครงสร้าง; signal ถูกบังคับค่า (ค่าอื่น = ล้มเหลว); ลองใหม่แล้วสำเร็จ; ล้มครบแล้ว raise พร้อมชื่อ node; template อ้าง key ที่ไม่มี; dry-run ชนะ `llm=` และ `ctx.llm`; context หาย → ไม่สร้าง LLM จริง
2. `loop_limit`: นับถูก; `again` / `exhausted` ถูกจังหวะ; `max_rounds=1` → exhausted ทันที
3. pattern ทั้งเส้น: ผ่านรอบแรก (execute 1 ครั้ง); revise หนึ่งครั้งแล้วผ่าน และ feedback ปรากฏในข้อความที่ส่งให้ executor รอบสอง; ครบรอบ → ทาง exhausted, `<id>__exhausted` เป็น True, execute รัน `max_rounds` ครั้งพอดี; planner รันครั้งเดียว; executor เป็น node ของผู้ใช้; สองชุดใน workflow เดียวนับรอบแยกกัน; ต่อ "exhausted" เข้า `approval_gate`
4. builder: signal ไม่ตรงของ `llm`/`loop_limit` → จบ run; `hitl` และชนิดอื่นคงเดิม
5. `advisor` หลังเปลี่ยนไปใช้ `resolve_llm`: test เดิมผ่านเท่าเดิม
6. `Fragment.exits`: `approval_gate` เดิมยังผ่านทุก test; `to_workflow()` ต่อทุกทางออก

## 9. ฝั่ง `lgtools`

contract test นับ node จาก `lgkit.nodes.*` แล้ว จึงต้องทำในรอบเดียวกัน:

- `NodeKind` เพิ่ม `'llm' | 'loop_limit'`; `NODE_COLORS` เพิ่มสองสี
- Inspector: panel ของ `llm` (message, result_key, output_fields, signal_field/signal_values, llm, max_attempts — ช่อง prompt แสดงอยู่แล้วผ่าน `uses_node_prompt`) และ `loop_limit` (max_rounds, counter_key)
- `register_all()` เรียก `ensure_registered()` อยู่แล้ว → palette มีสองชนิดใหม่ทันที
- dry-run / test case runner: ใช้ `mock_scripts[<node id>]` เดิม ไม่ต้องแก้ runner
- backend + frontend suite ต้องไม่ถอย

## 10. ความเสี่ยง / สิ่งที่ต้องยืนยันตอนทำแผน

- `FakeModel` ของ `lgtools` (`with_structured_output` คืนตัวเอง, `invoke` คืน `AIMessage`) — `llm` แบบข้อความธรรมดาต้องอ่าน `.content` ได้ทั้งจาก model จริงและ fake
- `events` ใน state เก็บเฉพาะ step ล่าสุด (พฤติกรรมเดิมของ `lgtools`) — test ของ pattern พิสูจน์เส้นทางด้วย `scratch` ไม่ใช่ `events`
- การสร้าง pydantic model จาก `output_fields` — ใช้ชนิดจาก `StructuredOutputField` ที่มีอยู่ใน `lgkit.spec` ถ้าใช้ซ้ำได้
