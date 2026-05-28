"""
Lightweight query rewriting for RAG retrieval.

Expands colloquial / ambiguous student questions into formal search
terms that match regulatory document wording.  Only triggers on
genuinely colloquial queries — formal questions pass through as-is.
"""

# Colloquial patterns that indicate a query needs rewriting
COLLOQUIAL_PATTERNS = [
    "咋办", "咋整", "咋搞", "能行吗", "可以不", "可以不", "ok吗",
    "啥时候", "咋弄", "咋申请", "咋退", "咋改",
    "能补吗", "能退吗", "能转吗", "能改吗",
    "挂科", "翘课", "旷课", "逃课",
    "行不行", "好不好", "可不可以",
    "那它", "那这", "这个呢", "那个呢",
]

COLLOQUIAL_SHORT_MAX = 6  # only rewrite very short/ambiguous fragments


def should_rewrite(question: str) -> bool:
    """Return True if *question* looks colloquial and worth rewriting."""
    text = question.strip()
    if len(text) <= COLLOQUIAL_SHORT_MAX:
        return True
    for pattern in COLLOQUIAL_PATTERNS:
        if pattern in text:
            return True
    return False


REWRITE_PROMPT = """你是一个高校教务查询助手。请把学生的口语化问题改写为适合检索校规文档的关键词。

规则：
1. 将口语化表达替换为正式用语：挂科→不及格 咋办→如何处理 能行吗→是否允许
2. 如果问题过于简短或有歧义（如"那它呢"），补全为完整的检索短语
3. 提取问题的核心实体和动作，去除礼貌用语和语气词
4. 不要回答问题，只输出改写后的关键词串（15字以内最佳）

输入：{question}
输出（仅输出改写后的查询，不要解释）："""


class QueryRewriter:
    """Rewrite colloquial student questions into retrieval-friendly queries.

    Only rewrites when ``should_rewrite()`` returns True — formal questions
    pass through unchanged to avoid degrading retrieval quality.
    """

    def __init__(self, llm_client):
        self._llm = llm_client

    def rewrite(self, question: str) -> str:
        """Return rewritten query, or the original if not needed / on failure."""
        if not should_rewrite(question):
            return question
        try:
            prompt = REWRITE_PROMPT.format(question=question)
            result = self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            rewritten = result.strip()
            if not rewritten or len(rewritten) < 2:
                return question
            return rewritten
        except Exception:
            return question
