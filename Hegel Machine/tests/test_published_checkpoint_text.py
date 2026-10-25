"""Published theory texts are reused without caching ordinary author files."""

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

from hegel_machine.cognition import checkpoint_storage as storage
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.anonymous_method_learning import compact_workspace_result


def test_index_and_library_publishers_share_their_exact_file_text(tmp_path, monkeypatch):
    agent = SimpleNamespace(workspace=tmp_path, files={}, questions=[], relation_research={},
        events=[], theory_index={'native': {'id': 'native', 'version': 1,
            'program': {'op': 'const', 'value': 2.}, 'note': '旧资料'}},
        theory_consolidation={'candidates': {}, 'executions': {}, 'transfers': []})
    consolidation.save_theory_index(agent)
    consolidation._save(agent)
    assert len(agent._published_checkpoint_text) == 2
    for name in (consolidation.INDEX_PATH, consolidation.LIBRARY_PATH):
        assert agent._published_checkpoint_text[tmp_path / name][1] is agent.files[name]

    def no_second_read(path, *args, **kwargs):
        raise AssertionError('a just-published theory file was read a second time')

    monkeypatch.setattr(Path, 'read_text', no_second_read)
    consolidation.save_theory_index(agent)
    consolidation._save(agent)
    for name in (consolidation.INDEX_PATH, consolidation.LIBRARY_PATH):
        assert storage.checkpoint_file_text(agent, tmp_path / name) is agent.files[name]


def test_published_text_detects_same_size_unicode_external_edit_with_restored_mtime(tmp_path):
    path = tmp_path / 'index.json'
    path.write_text('{"值":"甲"}', encoding='utf-8')
    agent = SimpleNamespace()
    original = storage.published_checkpoint_text(agent, path)
    before = path.stat()
    path.write_text('{"值":"乙"}', encoding='utf-8')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert storage.checkpoint_file_text(agent, path) == '{"值":"乙"}' != original
    assert path not in agent._published_checkpoint_text


def test_workspace_change_list_reuses_theories_but_always_reads_author_files(tmp_path, monkeypatch):
    names = ('shared/theory_index.json', 'shared/theory_library.json', 'model.py')
    texts = ('{"旧稿":"甲"}', '{"理论":"甲"}', '# 旧作者文件\n')
    agent = SimpleNamespace(workspace=tmp_path, files=dict(zip(names, texts)), questions=[],
        inherited_files=set(names), knowledge={'files': dict(zip(names, texts))})
    for name, text in zip(names, texts):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        if name != 'model.py':
            agent.files[name] = storage.published_checkpoint_text(agent, path)

    read = Path.read_text
    reads = []

    def counted(path, *args, **kwargs):
        reads.append(path)
        return read(path, *args, **kwargs)

    def result():
        return {'outgoing_knowledge': {'files': {}, 'question_traces': []},
                'online_questions': []}

    before = deepcopy(agent.knowledge)
    monkeypatch.setattr(Path, 'read_text', counted)
    assert compact_workspace_result(agent, result())['inherited_files_changed'] == []
    assert reads == [tmp_path / 'model.py']
    reads.clear()
    (tmp_path / names[1]).write_text('{"理论":"乙"}', encoding='utf-8')
    (tmp_path / 'model.py').write_text('# 新作者文件\n', encoding='utf-8')
    assert compact_workspace_result(agent, result())['inherited_files_changed'] == ['model.py', names[1]]
    assert set(reads) == {tmp_path / names[1], tmp_path / 'model.py'}
    assert agent.knowledge == before
