"""
Lightweight query rewriting for RAG retrieval.

Expands colloquial / ambiguous student questions into formal search
terms that match regulatory document wording.  Only triggers on
genuinely colloquial queries — formal questions pass through as-is.
"""

"""
Lightweight query rewriting for RAG retrieval.

Expands colloquial / ambiguous student questions into formal search
terms that match regulatory document wording.  Produces dual queries:
  - bm25_query: keyword expansion for BM25 (synonyms, formal terms)
  - vector_query: natural language for embedding search

Phase 2.2: relaxed guard — triggers on broader set of informal patterns.
"""

# Colloquial / informal patterns that indicate a query needs rewriting.
# Expanded from V1 to cover more student-speak patterns.
COLLOQUIAL_PATTERNS = [
    # Existing
    "咋办", "咋整", "咋搞", "能行吗", "可以不", "ok吗",
    "啥时候", "咋弄", "咋申请", "咋退", "咋改",
    "能补吗", "能退吗", "能转吗", "能改吗",
    "挂科", "翘课", "旷课", "逃课",
    "那它", "那这", "这个呢", "那个呢",
    "要多久", "要多长", "多长时间", "多久做完",
    "怎么弄", "怎么搞", "怎么做", "怎么申请",
    "要什么", "需要什么", "要哪些",
    "行不行", "好不好", "可不可以",
    # Phase 2.2 additions — broader informal coverage
    "怎么办", "要不要", "能不能", "会不会",
    "可以吗", "一样吗", "区别吗", "什么时候",
    "多少学分", "多少钱", "几次", "多久",
    "哪个", "哪些", "有什么", "没什么",
    "贵不贵", "难不难", "多不多",
    "需要什么", "流程是什么", "条件是什么",
]
COLLOQUIAL_SHORT_MAX = 8  # raise from 6 to catch more ambiguous short queries


def should_rewrite(question: str) -> bool:
    """Return True if *question* looks colloquial and worth rewriting."""
    text = question.strip()
    if len(text) <= COLLOQUIAL_SHORT_MAX:
        return True
    for pattern in COLLOQUIAL_PATTERNS:
        if pattern in text:
            return True
    return False


# Dual-purpose rewrite prompt: generates both BM25 keywords and a natural query
REWRITE_PROMPT = """你是一个高校教务查询助手。把学生的口语化问题改写为检索关键词。

规则：
1. 将口语化表达替换为正式用语，同时给出多个同义关键词：挂科→不及格 重修 补考 60分以下
2. 提取问题的核心实体（课程类型、流程名称、部门名称）
3. 输出格式：【关键词】空格分隔的检索词（15字以内）
4. 不要回答问题

输入：{question}
输出（仅输出关键词）："""


class QueryRewriter:
    """Rewrite colloquial questions into BM25-friendly keyword queries.

    Phase 2.2: relaxed guard catches more informal patterns.
    Rewrite now focuses on keyword expansion for BM25; vector search
    still uses the original question for semantic matching.
    """

    def __init__(self, llm_client, timeout: int = 5):
        self._llm = llm_client
        self._timeout = timeout

    def rewrite(self, question: str) -> str:
        """Return BM25-optimized keyword query, or original on failure/timeout."""
        if not should_rewrite(question):
            return question
        try:
            import threading
            result_container = []
            error_container = []

            def _call():
                try:
                    prompt = REWRITE_PROMPT.format(question=question)
                    r = self._llm.chat(
                        [{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    result_container.append(r.strip())
                except Exception as e:
                    error_container.append(e)

            thread = threading.Thread(target=_call, daemon=True)
            thread.start()
            thread.join(timeout=self._timeout)

            if result_container:
                rewritten = result_container[0]
                if rewritten and len(rewritten) >= 2:
                    return rewritten[:80]
            return question
        except Exception:
            return question
