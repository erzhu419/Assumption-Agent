"""Real Python output survives the bounded previews used in model context."""

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent


def test_large_python_streams_keep_opening_analysis_and_final_diagnostics(tmp_path):
    agent = InquiryLLMAgent(None, tmp_path, arm="code_react")
    source = (
        "import sys\n"
        "sys.stdout.write('first observation\\n' + 'x' * 13000 + '\\nlast observation\\n')\n"
        "sys.stderr.write('first diagnostic\\n' + 'y' * 14000 + '\\nlast diagnostic\\n')\n"
    )
    result = agent._execute("run", {"code": source}, None)

    assert result["returncode"] == 0 and not result["timeout"]
    assert result["executed_path"] == "scratch_000.py"
    for stream, first, last in (("stdout", "first observation", "last observation"),
                                ("stderr", "first diagnostic", "last diagnostic")):
        assert result[f"{stream}_file"] == f"scratch_000.py.{stream}.txt"
        complete = (tmp_path / result[f"{stream}_file"]).read_text()
        assert complete.startswith(first) and complete.endswith(last + "\n")
        assert len(result[stream]) == 12000 and result[stream].endswith(last + '\n')
        if stream == 'stdout':
            assert result[stream].startswith(first)
            assert 'middle omitted' in result[stream]
        else:
            assert result[stream] == complete[-12000:]
        assert result[f"{stream}_characters"] == len(complete)
        assert result[f"{stream}_truncated"] is True
    assert "beginning and end" in result["output_notice"]


def test_explicit_script_output_paths_and_short_rerun_replace_previous_streams(tmp_path):
    agent = InquiryLLMAgent(None, tmp_path, arm="hegel")
    path = "shared/inspect.py"
    agent._execute("write", {"path": path, "code": "print('old data' * 2000)\n"}, None)
    previous = agent._execute("run", {"path": path}, None)
    assert previous["stdout_truncated"] is True

    agent._execute("write", {"path": path, "code": "print('summary: 7 readings')\n"}, None)
    result = agent._execute("run", {"path": path}, None)

    assert result["executed_path"] == path
    assert result["returncode"] == 0 and not result["timeout"]
    assert result["stdout"] == "summary: 7 readings\n" and result["stderr"] == ""
    for stream in ("stdout", "stderr"):
        assert result[f"{stream}_file"] == f"{path}.{stream}.txt"
        assert result[f"{stream}_file"] == previous[f"{stream}_file"]
        assert (tmp_path / result[f"{stream}_file"]).read_text() == result[stream]
        assert result[f"{stream}_characters"] == len(result[stream])
        assert result[f"{stream}_truncated"] is False
    assert "output_notice" not in result
