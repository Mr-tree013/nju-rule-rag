"""Apply all Roadmap V3 B.2 + B.3 changes in one shot."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── 1. config.py: add enable_high_risk_deepseek + wire to factory ──
cfg = ROOT / "app/config.py"
c = cfg.read_text()

# Add field to dataclass (after max_answer_length)
c = c.replace(
    "max_answer_length: int = 600\n",
    "enable_high_risk_deepseek: bool = False\n    max_answer_length: int = 600\n"
)

# Add to create_settings() factory (after the existing max_answer_length line)
c = c.replace(
    '        max_answer_length=600,\n',
    '        max_answer_length=600,\n        enable_high_risk_deepseek=os.getenv("ENABLE_HIGH_RISK_DEEPSEEK", "false").lower() in ("true", "1", "yes"),\n'
)

cfg.write_text(c)
print("1. config.py: added enable_high_risk_deepseek")

# ── 2. pipeline.py: add high-risk routing ──
pipe = ROOT / "app/pipeline.py"
p = pipe.read_text()

# Add routing before the LLM call in answer()
old_block = """        # 9. Call LLM
        t0 = time.time()
        try:
            if self._settings.enable_two_stage_generation:
                answer_text = self._generate_two_stage(question, top_chunks)
            else:
                answer_text = self._generate(messages)"""

new_block = """        # 9. Call LLM (route high-risk to DeepSeek if enabled)
        t0 = time.time()
        try:
            if (self._settings.enable_high_risk_deepseek
                    and classification.level == RiskLevel.HIGH
                    and self._fallback_llm):
                answer_text = self._generate_with_fallback(messages)
            elif self._settings.enable_two_stage_generation:
                answer_text = self._generate_two_stage(question, top_chunks)
            else:
                answer_text = self._generate(messages)"""

p = p.replace(old_block, new_block)

# Add _generate_with_fallback helper
helper = """
    def _generate_with_fallback(self, messages: list[dict]) -> str:
        \"\"\"Route to DeepSeek fallback for high-risk questions.\"\"\"
        assert self._fallback_llm is not None
        self._llm_used = self._fallback_llm.model
        print(f"[LLM] 高风险题分流到 {self._fallback_llm.model}")
        try:
            return self._fallback_llm.chat(messages, temperature=0.15)
        except LLMError:
            print("[LLM] 回退模型也失败，使用主模型")
            self._llm_used = self._llm.model
            return self._llm.chat(messages, temperature=0.15)
"""
# Insert before the _format_response method
p = p.replace(
    "    def _format_response(",
    helper + "\n    def _format_response("
)

pipe.write_text(p)
print("2. pipeline.py: added high-risk routing to DeepSeek")

print("ALL DONE")
