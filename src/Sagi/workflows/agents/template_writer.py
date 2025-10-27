import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, UserMessage
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage

from resources.functions import get_hi_rag_client
from Sagi.workflows.sagi_memory import SagiMemory
from Sagi.vercel import ToolInputAvailable, ToolInputStart, ToolOutputAvailable


@dataclass
class PlanStep:
    module: str
    description: str


@dataclass
class Plan:
    steps: List[PlanStep]


class TemplateWriterAgent:
    """
    Lightweight pipeline (no AssistantAgent) for template-based writing:
      1) Extract template and instruction from user messages
      2) Plan per-module TODO list (emit Tool events)
      3) Run per-module RAG retrieval concurrently (emit Tool events)
      4) Stream final markdown generation following template structure
    """

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        model_client_stream: bool = True,
        markdown_output: bool = True,
    ):
        self.model_client = model_client
        self.memory = memory
        self.language = language
        self.model_client_stream = model_client_stream
        self.markdown_output = markdown_output

        self.template_text: str = ""
        self.user_instruction: str = ""
        self.plan: Optional[Plan] = None
        self.step_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.step_queries: Dict[str, str] = {}

        self._rag = get_hi_rag_client()

    # ------------------------- helpers -------------------------

    @staticmethod
    def _safe_json_extract(s: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        m = re.search(r"(\{[\s\S]*\})", s.strip())
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None

    @staticmethod
    def _extract_code_block(text: str) -> Tuple[str, str]:
        """
        Extract first code block as template and return (template, remaining_text_as_instruction).
        """
        m = re.search(r"```(?:markdown)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if not m:
            return "", text
        start, end = m.span()
        template = m.group(1).strip()
        remaining = (text[:start] + text[end:]).strip()
        return template, remaining

    @staticmethod
    def _join_messages(messages: List[TextMessage]) -> str:
        return "\n\n".join(m.content for m in messages if getattr(m, "content", ""))

    def _build_plan_prompt(self, template: str, instruction: str) -> str:
        lang = (self.language or "en").lower()
        if lang.startswith("cn"):
            return f"""
你是一个结构化规划助手。根据模板内容与用户指令，输出用于并行执行的 TODO 列表。
- 从模板中抽取模块顺序（例如 A, B, C, D），保留原始结构与顺序。
- 每个模块生成一条步骤，字段必须是 module 与 description。
- 严格输出 JSON，格式：
{{
  "steps": [{{"module": "模块名", "description": "该模块需要完成的目标/约束/产出"}}]
}}

[模板]
```
{template[:8000]}
```

[用户指令]
{instruction[:4000]}

只输出 JSON，不要多余文本。
""".strip()
        else:
            return f"""
You are a structured planning assistant. Given a template and user instruction, output a TODO list for parallel execution.
- Derive module names in order from the template (e.g., A, B, C, D), preserving structure and order.
- For each module, produce a step with fields: module and description.
- Output strictly valid JSON:
{{
  "steps": [{{"module": "ModuleName", "description": "Goal/constraints/deliverable for this module"}}]
}}

[Template]
```
{template[:8000]}
```

[Instruction]
{instruction[:4000]}

Output JSON only, with no extra text.
""".strip()

    def _build_generation_messages(self) -> List[UserMessage]:
        per_module_ctx: List[str] = []
        for module, chunks in self.step_chunks.items():
            snippet_texts: List[str] = []
            for c in chunks[:8]:
                t = (c.get("text") or "").strip()
                if t:
                    snippet_texts.append(t[:800])
            block = f"## {module}\n" + ("\n".join(f"- {s}" for s in snippet_texts) if snippet_texts else "- (no retrieval)")
            per_module_ctx.append(block)
        ctx = "\n\n".join(per_module_ctx)

        # Build plan overview to guide headings and structure
        plan_lines: List[str] = []
        plan_json_lines: List[str] = []
        if self.plan and self.plan.steps:
            for step in self.plan.steps:
                plan_lines.append(f"- {step.module}: {step.description}")
                # lightweight JSON-ish to guide model alignment
                plan_json_lines.append(
                    "  {\"module\": \"" + step.module.replace("\"", "'") + "\", \"required_h2_from\": \"" + step.description.replace("\"", "'") + "\"}"
                )
        plan_block = "\n".join(plan_lines) if plan_lines else "(none)"
        plan_json_block = (
            "[\n" + ",\n".join(plan_json_lines) + "\n]" if plan_json_lines else "[]"
        )
        module_queries_lines: List[str] = []
        for m, q in self.step_queries.items():
            module_queries_lines.append(f"- {m}: {q}")
        module_queries_block = "\n".join(module_queries_lines)

        lang = (self.language or "en").lower()
        if lang.startswith("cn"):
            sys = UserMessage(
                source="system",
                content="""
你是一个文档生成助手。请基于模板结构、计划步骤与每个模块的检索上下文，生成最终 Markdown 文档。
- 开头输出 `filename: 文件名.md`
- 紧接着用一个 ```markdown 代码块输出完整文档
- 严格遵循模板的模块顺序与层级（例如模板中是 # A / # B / # C）
- 严格对齐计划：对 MODULE-SPEC 中的每一项，输出对应模块的内容；不得增删模块；顺序必须一致
- 每个模块下只允许 1 个二级标题（## …），且该标题必须“根据计划中的 required_h2_from 改写生成”；不得复用检索片段原始标题；不同模块不得出现相同的二级标题
- 对检索内容进行“去重、合并与改写”，避免重复段落与重复句式；不要逐字拷贝检索中的段落标题（例如不要重复使用 “A Very General Overview of …” 作为小节标题）
- 每个模块建议 1 个二级标题 + 若干要点句（2-5 句），必要时可用列表；如检索为空，则依据指令进行合理补全
- 内容应简洁、通顺，避免堆砌与冗余
<输出契约>
- 对于 MODULE-SPEC 中的每个对象 {module, required_h2_from}，按以下结构输出：
  # {module}\n
  ## {改写自 required_h2_from 的独特小节标题}\n
  {2-5 句要点或简短列表}
- 不要输出其他多余的二级标题或额外模块
""".strip()
            )
        else:
            sys = UserMessage(
                source="system",
                content="""
You are a document generator. Using the template structure, the plan steps, and per-module retrieved context, produce a final Markdown document.
- Start with `filename: <file_name>.md`
- Then output a single ```markdown fenced block with the full document
- Strictly follow the template module order and hierarchy (e.g., # A / # B / # C)
- STRICT ALIGNMENT: For each MODULE-SPEC item, output the corresponding module content; do not add/remove modules; preserve order
- Exactly one H2 (## …) per module and it MUST be uniquely paraphrased from `required_h2_from`; do NOT reuse raw titles from retrieved chunks; different modules must not share the same H2
- Deduplicate, merge, and paraphrase retrieved text; avoid copy-pasting source headings (e.g., do not repeat “A Very General Overview of …”)
- Aim for 1 H2 per module + 2–5 concise sentences (or a short list). If no retrieval, reasonably complete the content from the instruction
- Keep it concise and coherent; avoid redundancy
<OUTPUT CONTRACT>
- For each object {module, required_h2_from} in MODULE-SPEC, output:
  # {module}\n
  ## {unique H2 paraphrased from required_h2_from}\n
  {2–5 sentences or a short list}
- Do not output extra H2s or extra modules
""".strip()
            )

        user = UserMessage(
            source="user",
            content=f"""
[TEMPLATE]
```
{self.template_text[:8000]}
```

[INSTRUCTION]
{self.user_instruction[:4000]}

[MODULE-SPEC]
```json
{plan_json_block}
```

[MODULE-QUERIES]
{module_queries_block}

[PLAN]
{plan_block}

[PER-MODULE CONTEXT]
{ctx[:12000]}
""".strip()
        )
        return [sys, user]

    async def _prepare_inputs_from_messages(
        self, messages: List[TextMessage]
    ) -> Tuple[str, str]:
        full_text = self._join_messages(messages)
        template, instruction = self._extract_code_block(full_text)
        self.template_text = template
        self.user_instruction = instruction
        return template, instruction

    # ------------------------- public streaming APIs -------------------------

    async def run_plan(
        self,
        messages: List[TextMessage],
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        template, instruction = await self._prepare_inputs_from_messages(messages)

        yield ToolInputStart(toolName="templatePlan")
        yield ToolInputAvailable(
            input={
                "type": "templatePlan-input",
                "instruction": (self.user_instruction or "")[:2000],
                "templatePreview": (self.template_text or "")[:2000],
            }
        )

        prompt = self._build_plan_prompt(template, instruction)
        res = await self.model_client.create(
            [UserMessage(content=prompt, source="user")], cancellation_token=cancellation_token
        )
        content = (res.content or "").strip()

        plan_json = self._safe_json_extract(content) or {"steps": []}
        steps: List[PlanStep] = []
        for s in plan_json.get("steps", []):
            module = (s.get("module") or "").strip()
            desc = (s.get("description") or "").strip()
            if module:
                steps.append(PlanStep(module=module, description=desc))
        self.plan = Plan(steps=steps)

        yield ToolOutputAvailable(
            output={"type": "templatePlan-output", "data": {"steps": [s.__dict__ for s in steps]}}
        )

    async def run_steps_retrieval(
        self,
        *,
        workspace_id: str,
        knowledge_base_id: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        if not self.plan:
            return

        await self._rag.set_language(self.language)
        self.step_chunks = {}
        self.step_queries = {}

        queue: asyncio.Queue = asyncio.Queue()
        done_sentinel = object()
        total = len(self.plan.steps)

        async def worker(step: PlanStep):
            await queue.put(ToolInputStart(toolName="ragSearch"))
            await queue.put(
                ToolInputAvailable(
                    input={
                        "type": "ragSearch-input",
                        "module": step.module,
                        "query": step.description or step.module,
                    }
                )
            )

            query_text = step.description or step.module
            self.step_queries[step.module] = query_text
            rag_task = asyncio.create_task(
                self._rag.query(
                    query_text,
                    workspace_id=workspace_id,
                    knowledge_base_id=knowledge_base_id,
                    translation=["en", "zh", "zh-t-hk"],
                    summary=False,
                    filter_by_clustering=False,
                )
            )
            if cancellation_token is not None:
                cancellation_token.link_future(rag_task)
            try:
                ret = await rag_task
            except asyncio.CancelledError:
                rag_task.cancel()
                raise

            chunks = ret.get("chunks", []) or []
            self.step_chunks[step.module] = chunks

            items = list(
                {
                    "fileName": c.get("fileName"),
                    "fileUrl": c.get("uri"),
                    "type": (c.get("uri") or "").split(".")[-1],
                }
                for c in chunks[:12]
            )

            await queue.put(
                ToolOutputAvailable(
                    output={"type": "ragSearch-output", "module": step.module, "data": items}
                )
            )
            await queue.put(done_sentinel)

        tasks = [asyncio.create_task(worker(s)) for s in self.plan.steps]
        finished = 0
        try:
            while finished < total:
                evt = await queue.get()
                if evt is done_sentinel:
                    finished += 1
                else:
                    yield evt
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    def run_generate(
        self,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        messages = self._build_generation_messages()

        async def _stream():
            async for chunk in self.model_client.create_stream(
                messages, cancellation_token=cancellation_token
            ):
                if isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(content=chunk, source="assistant")

        return _stream()


