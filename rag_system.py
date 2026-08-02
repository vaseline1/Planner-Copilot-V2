"""rag_system.py —— 招生简章 PDF 问答系统（RAG）

职责：
    1. 解析 PDF、按"段落 → 句子 → 硬切兜底"三级策略分块（带重叠窗口）
    2. 用 SiliconFlow bge-large-zh-v1.5 把文本块向量化，存进 pgvector
    3. 规划师提问时，把问题也向量化，检索最相关的文本块
    4. 把检索到的原文喂给 DeepSeek，让它"只根据这些原文回答"，并标注来源

为什么这是必要的一层（而不是只用 knowledge_base.py 的关键词检索）：
    "南开有没有强基"是关键词检索就能答的问题；"南开强基校测成绩占综合
    成绩的百分之多少"这种问题，答案往往藏在简章某一条款的具体表述里，
    只有让模型"看到原文再回答"才不会瞎编——这正是 RAG 的核心价值。

分块引擎说明（本版核心变更）：
    parse_pdf() 内部已替换为 chunk_pdf.py 的完整分块引擎，取代旧版
    "按 \\n\\n 切段 + 250 字符硬切"的简单实现。新引擎解决了三个老问题：
      ① 硬切不看标点，会把一句话/一个词从中间切断 —— 现在句子边界优先，
         字符硬切只作为"单句仍超限"时的最后兜底；
      ② 旧版只要 page.get_text() 返回空就静默 continue，混合文档
         （部分页文字、部分页扫描图）丢了页都不知道 —— 现在逐页独立判断，
         无文字层的页记 warning 日志并计数汇报（本系统不做 OCR）；
      ③ 页码、纯数字行、"3/15"这类碎片会被当成正文向量化 —— 现在按
         噪声模式直接过滤。
    对外的返回结构不变：[{"page": int, "content": str}, ...]，
    index_pdf() 一行都不用改。

⚠️ 这个文件的代码本身是完整、可运行的，但"喂给它什么 PDF"是你的工作：
    我不会替你生产真实的招生简章内容。index_pdf() 可以处理任何你自己
    下载/整理好的官方简章 PDF（文字版；扫描件请先自行 OCR）。
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
from openai import OpenAI
from sqlmodel import Session, text

from config import RAG_SIMILARITY_THRESHOLD, RAG_TOP_K, get_chat_config, get_embedding_config
from database import engine
from observability import observe, update_current_generation, update_current_span
from prompts import RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 分块参数（移植自 chunk_pdf.py 的默认值；调用 parse_pdf 时可显式传参覆盖）
# ══════════════════════════════════════════════════════════════

RAG_CHUNK_SIZE = 300    # 每块目标字数
RAG_CHUNK_OVERLAP = 50  # 相邻块重叠字数（保证跨块的句子不断裂）
RAG_MIN_CHARS = 10      # 段落/碎块最小保留字符数


# ══════════════════════════════════════════════════════════════
# 分块引擎（移植自 chunk_pdf.py：句子边界优先 + 重叠窗口 + 噪声过滤）
# ══════════════════════════════════════════════════════════════

# 句末标点（中/英文）及其后可能跟随的闭引号、闭括号 —— 用来找"句子边界"
_SENTENCE_END = re.compile(r'([。！？!?]+[”’"\'）)]*)')

# 噪声碎片：纯数字行、纯符号行、"第3页/第2章"、"3/15"、"3 of 15" 这类页码
_NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*[-—–_.…=*#~·•\s]+$"),
    re.compile(r"^\s*第?\s*[0-9一二三四五六七八九十]+\s*[页章]\s*$"),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*\d+\s*of\s*\d+\s*$", re.I),
]


def _is_noise_fragment(text: str) -> bool:
    """判断一段文本是不是"页码/分隔线"这类不值得向量化的噪声碎片。"""
    t = text.strip()
    if not t:
        return True
    return any(p.match(t) for p in _NOISE_PATTERNS)


def _split_into_sentences(text: str) -> list[str]:
    """按句末标点切句子。re.split 会把分隔符（标点）也作为独立元素返回，
    所以用 buf 把"正文 + 它自己的句末标点"重新拼回一个完整句子。"""
    parts = re.split(_SENTENCE_END, text)
    sentences: list[str] = []
    buf = ""
    for part in parts:
        buf += part
        if _SENTENCE_END.fullmatch(part):
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return [s for s in sentences if s]


def _hard_slice(text: str, size: int) -> list[str]:
    """字符硬切——只在"单句仍然超过 chunk_size"时兜底使用
    （比如无标点的长串数字、英文长段）。"""
    size = max(size, 1)
    return [text[i:i + size] for i in range(0, len(text), size)] or [text]


def _append_with_overlap(prev: str, unit: str, chunk_size: int, overlap: int) -> str:
    """开新块时，把上一块的尾部 overlap 个字接到新块开头——保证跨块
    边界的句子在检索时仍然完整可读。放不下（unit 本身就快占满）就不重叠。"""
    if not prev or overlap <= 0:
        return unit
    room = chunk_size - len(unit)
    if room <= 0:
        return unit
    tail_len = min(overlap, room, len(prev))
    return prev[-tail_len:] + unit


def _chunk_text_list(
    paragraphs: list[str],
    chunk_size: int = RAG_CHUNK_SIZE,
    overlap: int = RAG_CHUNK_OVERLAP,
    min_chars: int = RAG_MIN_CHARS,
) -> list[str]:
    """核心分块：段落 →（超限时）句子 →（单句仍超限时）硬切，再按
    chunk_size 贪心装块，块间保留 overlap 字重叠。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 需要满足 0 <= overlap < chunk_size")

    # 第一步：把段落压成"不超过 chunk_size 的语义单元"列表
    units: list[str] = []
    for para in paragraphs:
        para = re.sub(r"\s+", " ", para).strip()
        if not para or len(para) < min_chars or _is_noise_fragment(para):
            continue
        if len(para) <= chunk_size:
            units.append(para)
            continue
        for sent in _split_into_sentences(para):
            if len(sent) <= chunk_size:
                units.append(sent)
            else:
                units.extend(_hard_slice(sent, chunk_size - overlap))

    # 第二步：贪心装块——能拼就拼，拼不下就封存当前块、带重叠开新块
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + len(unit) <= chunk_size:
            current += unit
        else:
            chunks.append(current)
            current = _append_with_overlap(current, unit, chunk_size, overlap)
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if len(c.strip()) >= min_chars]


def _split_to_paragraphs(page_text: str) -> list[str]:
    """将单页文本切成段落候选。

    PyMuPDF 的 text 模式几乎不保留空行（实测正常排版的 PDF 提取出来
    "\\n\\n" 出现次数也是 0），所以对 split("\\n\\n") 做了兜底：切不出
    多于 1 个块时，整页当一条连续文本交给后面的句子级分块处理，长句
    跨行才不会被行边界意外打断。真的切出多个空行块时，再用"连续短行
    且数量>2"识别列表/条目类内容（保留逐行独立，不强行合并）。
    """
    raw_blocks = [b.strip() for b in page_text.split("\n\n") if b.strip()]
    if len(raw_blocks) <= 1:
        return [page_text.strip()] if page_text.strip() else []

    paragraphs: list[str] = []
    for block in raw_blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) > 2 and all(len(line) < 40 for line in lines):
            paragraphs.extend(lines)
        else:
            merged = " ".join(lines)
            if merged:
                paragraphs.append(merged)
    return paragraphs


def parse_pdf(
    pdf_path: str,
    chunk_size: int = RAG_CHUNK_SIZE,
    overlap: int = RAG_CHUNK_OVERLAP,
    min_chars: int = RAG_MIN_CHARS,
) -> list[dict]:
    """解析 PDF 并分块。返回 [{"page": int, "content": str}, ...]
    （index_pdf() 依赖这个返回结构，与旧版完全一致）。

    逐页独立判断有没有文字层：无文字层的页（扫描件/图片）记 warning
    日志并跳过，最后统一汇报跳过了几页——不会像旧版那样静默丢页。
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    doc = fitz.open(pdf_path)
    chunks: list[dict] = []
    skipped_pages = 0
    try:
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text("text").strip()
            if not page_text:
                skipped_pages += 1
                logger.warning("第%d页无文字层（可能是扫描件/图片），已跳过——本系统不做 OCR", page_num)
                continue
            paragraphs = _split_to_paragraphs(page_text)
            for piece in _chunk_text_list(paragraphs, chunk_size, overlap, min_chars):
                chunks.append({"page": page_num, "content": piece})
    finally:
        doc.close()

    if skipped_pages:
        logger.warning(
            "%s 共有 %d 页因无文字层被跳过（如需索引这些页，请先 OCR 转成文字版 PDF）",
            pdf_path, skipped_pages,
        )
    return chunks


# ══════════════════════════════════════════════════════════════
# 向量化 + 入库 + 检索 + 问答
# ══════════════════════════════════════════════════════════════

def init_rag_db() -> None:
    """初始化 RAG 所需的数据库表（需要先手动执行一次）。

    调用前提：PostgreSQL 已启用 pgvector 扩展（`CREATE EXTENSION vector;`，
    check_setup.py 会检查这一项）。

    ⚠️ 关于索引——这里特意不建 ivfflat 近似索引：
        ivfflat 是"近似"最近邻索引，需要数据量和 lists 参数匹配才可靠。
        个人知识库场景下文本块通常只有几百到几千条，直接顺序扫描
        （PostgreSQL 精确计算每一行的余弦距离）又准又够快，没必要为了
        用不上的性能收益引入近似检索的风险——实测过：3条数据配
        lists=100 的 ivfflat 索引，检索时会直接漏掉本该排第一的结果
        （近似索引在数据量远小于 lists 时会失效，且不会报错，只是
        悄悄返回错误/不全的结果，非常隐蔽）。
        等文本块数量真的涨到几万条以上、顺序扫描明显变慢时，再按
        pgvector 官方文档建议加索引：`lists` 设成"行数/1000"这个量级，
        且查询前 `SET ivfflat.probes = 10;`（默认 probes=1 召回率偏低）。
    """
    emb_cfg = get_embedding_config()
    init_sql = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS document_chunks (
        id          SERIAL PRIMARY KEY,
        doc_id      TEXT NOT NULL,
        university  TEXT NOT NULL,
        path_type   TEXT NOT NULL,
        source_year INTEGER NOT NULL,
        page_num    INTEGER,
        chunk_index INTEGER NOT NULL,
        content     TEXT NOT NULL,
        embedding   vector({emb_cfg.dimension}),
        created_at  TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_university ON document_chunks(university);
    """
    with Session(engine) as session:
        session.exec(text(init_sql))
        session.commit()
    logger.info("RAG 数据库初始化完成（pgvector 扩展 + document_chunks 表已就绪，暂不建近似索引）")


def get_embedding(text_input: str) -> list[float]:
    """获取文本的向量表示。⚠️ 用 SiliconFlow bge-large-zh-v1.5，不是 DeepSeek。"""
    emb_cfg = get_embedding_config()
    client = OpenAI(api_key=emb_cfg.api_key, base_url=emb_cfg.base_url)
    response = client.embeddings.create(model=emb_cfg.model, input=text_input)
    return response.data[0].embedding


def index_pdf(pdf_path: str, university: str, path_type: str, source_year: int) -> int:
    """把一份招生简章 PDF 向量化并存入数据库。返回成功存储的文本块数量。"""
    doc_id = hashlib.md5(f"{university}{path_type}{source_year}".encode()).hexdigest()
    chunks = parse_pdf(pdf_path)
    if not chunks:
        logger.warning("PDF 解析出 0 个文本块（可能是扫描件，需要先 OCR）：%s", pdf_path)
        return 0

    stored_count = 0
    with Session(engine) as session:
        session.exec(text("DELETE FROM document_chunks WHERE doc_id = :doc_id"), params={"doc_id": doc_id})

        insert_sql = text(
            """
            INSERT INTO document_chunks
                (doc_id, university, path_type, source_year, page_num, chunk_index, content, embedding)
            VALUES
                (:doc_id, :university, :path_type, :source_year, :page, :idx, :content, :embedding)
            """
        )
        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk["content"])
            session.exec(
                insert_sql,
                params={
                    "doc_id": doc_id,
                    "university": university,
                    "path_type": path_type,
                    "source_year": source_year,
                    "page": chunk["page"],
                    "idx": idx,
                    "content": chunk["content"],
                    "embedding": str(embedding),
                },
            )
            stored_count += 1
        session.commit()

    logger.info("%s %s 简章已索引：%d 个文本块", university, path_type, stored_count)
    return stored_count


def _prefer_latest_year(chunks: list[dict], top_k: int) -> list[dict]:
    """同一（院校, 路径）组内只保留最新年份的文本块，再按原相似度顺序取 top_k。

    为什么需要这一步：同一所学校同一路径、不同年份的简章会共存于向量库
    （doc_id 按"学校+路径+年份"生成，新年简章不会覆盖旧年的）。如果检索
    不看年份，旧年条款可能因为措辞更接近问题而排在新年条款前面，大模型
    就会拿着旧简章回答。这里按"组内最新年份优先"过滤——学校某路径一旦
    录入了新一年的简章，旧年份的块自动不再进入回答上下文（旧数据仍保留
    在库里可追溯，只是不再被优先检索）。
    chunks 输入时已按向量距离升序（相似度降序）排列，过滤保持这个顺序。
    """
    latest_year: dict[tuple, int] = {}
    for c in chunks:
        key = (c["university"], c["path_type"])
        latest_year[key] = max(latest_year.get(key, 0), c["source_year"])
    fresh = [c for c in chunks if c["source_year"] == latest_year[(c["university"], c["path_type"])]]
    return fresh[:top_k]


@observe(as_type="retriever", name="retrieve_relevant_chunks")
def retrieve_relevant_chunks(
    query: str, university: Optional[str] = None, path_type: Optional[str] = None, top_k: int = RAG_TOP_K
) -> list[dict]:
    """检索与查询最相关的文本块（用 pgvector 的余弦距离排序 + 最新年份优先）。

    标成 as_type="retriever"：Langfuse 会把这类 span 在界面上单独归类，
    方便你事后筛选"这次对话到底检索到了什么、相似度排第几"，跟"检索
    没检索到东西"这种问题排查直接相关。
    """
    query_embedding = get_embedding(query)
    query_vec_str = str(query_embedding)

    # 先取出比 top_k 更多的候选（默认4倍），给"最新年份优先"过滤留出余量——
    # 如果只取 top_k 条再过滤掉旧年份，可能一条都不剩。
    fetch_k = top_k * 4

    filters = ["1=1"]
    params: dict = {"query_vec": query_vec_str, "fetch_k": fetch_k}
    if university:
        filters.append("university = :university")
        params["university"] = university
    if path_type:
        filters.append("path_type = :path_type")
        params["path_type"] = path_type
    where_clause = " AND ".join(filters)

    sql = text(
        f"""
        SELECT university, path_type, source_year, page_num, content,
               1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
        FROM document_chunks
        WHERE {where_clause}
        ORDER BY embedding <=> CAST(:query_vec AS vector)
        LIMIT :fetch_k
        """
    )
    with Session(engine) as session:
        rows = session.exec(sql, params=params).fetchall()

    candidates = [
        {
            "university": r.university,
            "path_type": r.path_type,
            "source_year": r.source_year,
            "page_num": r.page_num,
            "content": r.content,
            "similarity": round(float(r.similarity), 4),
        }
        for r in rows
        if r.similarity > RAG_SIMILARITY_THRESHOLD
    ]
    results = _prefer_latest_year(candidates, top_k)
    update_current_span(
        output={"命中数量": len(results), "最高相似度": results[0]["similarity"] if results else None}
    )
    return results


@observe(as_type="generation", name="rag_generation")
def _generate_answer_from_context(question: str, context: str):
    """单独抽出生成这一步，标成 as_type="generation"——原因和
    agent_core.py 的 _call_llm 一样：只有这个类型 Langfuse 才会把
    usage_details 换算成 Token 用量和费用展示出来。"""
    chat_cfg = get_chat_config()
    chat_client = OpenAI(api_key=chat_cfg.api_key, base_url=chat_cfg.base_url)
    response = chat_client.chat.completions.create(
        model=chat_cfg.model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"【参考资料】\n{context}\n\n【规划师问题】{question}"},
        ],
        max_tokens=1000,
    )
    if response.usage:
        update_current_generation(model=response.model, usage_details=response.usage.model_dump(exclude_none=True))
    return response


@observe(as_type="chain", name="rag_answer_question")
def answer_question(question: str, university: Optional[str] = None, path_type: Optional[str] = None) -> str:
    """RAG 问答主函数：检索 + 生成，任何一步失败都返回可读的降级提示，不抛异常。

    标成 as_type="chain"：这是"检索"和"生成"两个子步骤的编排者，Langfuse
    里会看到一条 chain span 下面挂着一条 retriever 子span、一条
    generation 子span——和 agent_core.py 里"一次对话下面挂着若干次工具
    调用"是同一种"父子 span"结构，看效果时两边体验是一致的。
    """
    try:
        chunks = retrieve_relevant_chunks(question, university, path_type)
    except Exception as exc:  # noqa: BLE001 —— 检索失败不能让整个 Agent 崩溃
        logger.warning("RAG 检索失败：%s", exc)
        return "⚠ 简章检索服务暂时不可用，建议规划师直接查阅官方简章。"

    if not chunks:
        return (
            "⚠ 未在已索引的简章中找到相关内容。\n"
            "可能原因：①该院校简章尚未上传 ②问题中的关键词与简章用词不一致\n"
            "建议：直接查阅该院校官网的最新招生简章，或用 index_pdf() 先索引对应简章。"
        )

    context = "\n\n".join(
        f"【资料{i}】来源：{c['university']} {c['path_type']} {c['source_year']}年简章 第{c['page_num']}页\n"
        f"内容：{c['content']}"
        for i, c in enumerate(chunks, 1)
    )

    try:
        response = _generate_answer_from_context(question, context)
        return response.choices[0].message.content or "（模型没有返回内容）"
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG 生成回答失败：%s", exc)
        return f"⚠ 已检索到 {len(chunks)} 条相关简章内容，但生成回答时出错：{exc}"


# ══════════════════════════════════════════════════════════════
# 文档库管理（网页"数据管理"页 / api.py RAG 端点使用）
# 每份简章 = 同一个 doc_id 下的若干文本块；管理粒度就是"整份文档"。
# ══════════════════════════════════════════════════════════════


def list_documents() -> list[dict]:
    """列出向量库里已索引的全部简章（按 doc_id 聚合）。

    返回 [{doc_id, university, path_type, source_year, chunk_count}, ...]
    """
    sql = text(
        """
        SELECT doc_id, university, path_type, source_year, COUNT(*) AS chunk_count
        FROM document_chunks
        GROUP BY doc_id, university, path_type, source_year
        ORDER BY source_year DESC, university, path_type
        """
    )
    with Session(engine) as session:
        rows = session.exec(sql).all()
    return [dict(row._mapping) for row in rows]


def delete_document(doc_id: str) -> int:
    """删除某份简章的全部文本块，返回删除的块数（0 = 文档不存在）。"""
    with Session(engine) as session:
        result = session.exec(
            text("DELETE FROM document_chunks WHERE doc_id = :doc_id"), params={"doc_id": doc_id}
        )
        session.commit()
        deleted = result.rowcount
    logger.info("已删除文档 %s 的 %d 个文本块", doc_id, deleted)
    return deleted
