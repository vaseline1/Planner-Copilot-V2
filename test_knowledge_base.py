"""test_knowledge_base.py —— knowledge_base.py 的单元测试

只需要本地 data/tianjin_programs.json 文件，不需要数据库/Redis/API Key。
"""
from knowledge_base import get_tianjin_summary, load_programs, search_programs


class TestLoadPrograms:
    def test_loads_sample_data(self) -> None:
        programs = load_programs()
        assert len(programs) >= 3, "示例数据应至少有3条（原方案自带的示例）"

    def test_missing_file_returns_empty_not_error(self) -> None:
        assert load_programs(path="data/不存在的文件.json") == []


class TestSearchPrograms:
    def test_search_by_university_name(self) -> None:
        results = search_programs("南开")
        assert len(results) >= 1
        assert all("南开" in r["院校"] for r in results)

    def test_search_by_path_keyword(self) -> None:
        results = search_programs("强基")
        assert len(results) >= 1

    def test_no_match_returns_empty(self) -> None:
        assert search_programs("一个不存在的关键词XYZ123") == []

    def test_every_result_has_disclaimer(self) -> None:
        """确保每条搜索结果都带着"请核查最新简章"的提示，不能让规划师误以为是最终数据。"""
        results = search_programs("南开")
        assert all("⚠ 提示" in r for r in results)


class TestTianjinSummary:
    def test_summary_structure(self) -> None:
        summary = get_tianjin_summary()
        assert "天津强基计划" in summary
        assert "天津综合评价" in summary
        assert "港澳升学" in summary
