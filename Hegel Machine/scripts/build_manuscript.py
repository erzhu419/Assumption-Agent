#!/usr/bin/env python3
"""Build this manuscript's Markdown, equations and figures into editable TeX/PDF."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlparse

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.regenerate_manuscript_tables import generate_tables

ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def escape(text):
    return "".join(ESCAPES.get(character, character) for character in text)


def math_inline(state, silent):
    if state.src[state.pos] != "$":
        return False
    end = state.src.find("$", state.pos + 1)
    if end < 0:
        return False
    if not silent:
        token = state.push("math_inline", "math", 0)
        token.content = state.src[state.pos + 1:end]
    state.pos = end + 1
    return True


def math_block(state, start, end, silent):
    first = state.bMarks[start] + state.tShift[start]
    if state.src[first:state.eMarks[start]].strip() != "$$":
        return False
    if silent:
        return True
    closing = start + 1
    while closing < end:
        line = state.src[state.bMarks[closing]:state.eMarks[closing]].strip()
        if line == "$$":
            break
        closing += 1
    if closing == end:
        raise ValueError(f"Unclosed display equation at Markdown line {start + 1}")
    token = state.push("math_block", "math", 0)
    token.content = state.getLines(start + 1, closing, 0, False).strip()
    token.map = [start, closing + 1]
    state.line = closing + 1
    return True


def parser():
    markdown = MarkdownIt("commonmark").enable("table")
    markdown.inline.ruler.before("escape", "math_inline", math_inline)
    markdown.block.ruler.before("paragraph", "math_block", math_block,
        {"alt": ["paragraph", "reference", "blockquote", "list"]})
    return markdown


class Manuscript:
    def __init__(self, source):
        self.source = source
        self.figures = []
        self.tables = 0
        self.equations = 0
        self.artifacts = {}

    def inline(self, children):
        result = []
        for token in children:
            if token.type == "text":
                result.append(escape(token.content))
            elif token.type in {"softbreak", "hardbreak"}:
                result.append("\n" if token.type == "softbreak" else r"\\" + "\n")
            elif token.type == "math_inline":
                result.append(r"\(" + token.content + r"\)")
            elif token.type == "strong_open":
                result.append(r"\textbf{")
            elif token.type == "em_open":
                result.append(r"\emph{")
            elif token.type in {"strong_close", "em_close", "link_close"}:
                result.append("}")
            elif token.type == "code_inline":
                result.append(r"\path{" + token.content + "}")
            elif token.type == "link_open":
                target = token.attrGet("href")
                if not urlparse(target).scheme and not target.startswith("#"):
                    path = (self.source.parent / unquote(target.split("#")[0])).resolve()
                    if not path.exists():
                        raise FileNotFoundError(path)
                    relative = path.relative_to(ROOT).as_posix()
                    number = self.artifacts.setdefault(relative, len(self.artifacts) + 1)
                    result.append(r"\hyperlink{artifact-" + str(number) + "}{")
                else:
                    target = target.replace("%", r"\%").replace("#", r"\#").replace("&", r"\&")
                    result.append(r"\href{" + target + "}{")
            else:
                raise ValueError(f"Unsupported manuscript inline token: {token.type}")
        return "".join(result)

    def figure(self, token, caption):
        relative = Path(token.attrGet("src"))
        vector = relative.with_suffix(".pdf")
        if (self.source.parent / vector).is_file():
            relative = vector
        if not (self.source.parent / relative).is_file():
            raise FileNotFoundError(self.source.parent / relative)
        self.figures.append(str(relative))
        return ("\\begin{figure}[htbp]\n\\centering\n"
            "\\includegraphics[width=\\linewidth,height=.78\\textheight,keepaspectratio]{"
            + relative.as_posix() + "}\n\\caption*{" + caption + "}\n\\end{figure}\n")

    def table(self, tokens, caption=None):
        rows, row, alignments = [], [], []
        header = True
        for token in tokens:
            if token.type == "tr_open":
                row = []
            elif token.type == "inline":
                cell = self.inline(token.children)
                cell = cell.replace("Instrument/coordinate", r"Instrument/\allowbreak coordinate")
                if header:
                    cell = cell.replace("Observations", r"Obser\-vations")
                    cell = cell.replace("Measurements", r"Measure\-ments")
                row.append(cell)
            elif token.type == "th_open":
                alignments.append("raggedleft" if "right" in (token.attrGet("style") or "") else "raggedright")
            elif token.type == "thead_close":
                header = False
            elif token.type == "tr_close":
                rows.append((header, row))
        count = len(rows[0][1])
        first = .30 if count == 2 else .52 if count == 3 else .23 if count == 6 else .28
        widths = [first] + [(1 - first) / (count - 1)] * (count - 1)
        if caption and count == 3:
            for prefix, layout in ((r"\textbf{Table 1.", [.28, .32, .40]),
                                   (r"\textbf{Table 2.", [.27, .37, .36]),
                                   (r"\textbf{Table 5.", [.35, .40, .25]),
                                   (r"\textbf{Table B1.", [.25, .42, .33])):
                if caption.startswith(prefix):
                    widths = layout
        columns = "@{}" + "".join(
            f">{{\\{alignments[index]}\\arraybackslash}}p{{\\dimexpr "
            f"{width:.5f}\\linewidth-{2 * (count - 1) / count:.5f}\\tabcolsep\\relax}}"
            for index, width in enumerate(widths)) + "@{}"
        head = " & ".join(r"\textbf{" + cell + "}" for cell in rows[0][1]) + r" \\" + "\n"
        result = [r"\begingroup\small\setlength{\tabcolsep}{4pt}" + "\n",
            r"\renewcommand{\arraystretch}{1.18}" + "\n"]
        compact_table = bool(caption and caption.startswith(
            (r"\textbf{Table 6.", r"\textbf{Table 8.", r"\textbf{Table F1.", r"\textbf{Table B2.")))
        if len(rows) <= 4 or compact_table:
            # Keep this manuscript's two/three-row comparisons and captions
            # together, including recall outcomes and the control/learning map.
            # Longer protocol tables retain ordinary page breaks.
            result.append(r"\par\addvspace{.7em}\noindent\begin{minipage}{\linewidth}" + "\n")
            if caption:
                result.append(r"\captionof*{table}{" + caption + "}\n")
            result.extend([r"\begin{tabular}{" + columns + "}\n",
                r"\toprule" + "\n", head, r"\midrule" + "\n"])
            result.extend(" & ".join(row) + r" \\" + "\n" for _, row in rows[1:])
            result.append("\\bottomrule\n\\end{tabular}\n\\end{minipage}"
                "\\par\\addvspace{.7em}\n\\endgroup\n")
            self.tables += 1
            return "".join(result)
        result.append(r"\begin{longtable}{" + columns + "}\n")
        if caption:
            result.append(r"\caption*{" + caption + r"}\\" + "\n")
        result.extend([r"\toprule" + "\n", head, r"\midrule\endfirsthead" + "\n",
                       r"\toprule" + "\n", head, r"\midrule\endhead" + "\n",
                       r"\bottomrule\endlastfoot" + "\n"])
        result.extend(" & ".join(row) + r" \\" + "\n" for _, row in rows[1:])
        result.append("\\end{longtable}\n\\endgroup\n")
        self.tables += 1
        return "".join(result)

    def render(self, text):
        tokens = parser().parse(text)
        output, index, reference_group = [], 0, False
        in_abstract = False
        while index < len(tokens):
            token = tokens[index]
            if token.type == "heading_open":
                title = self.inline(tokens[index + 1].children)
                plain = tokens[index + 1].content
                if token.tag == "h1":
                    output.append(r"\title{" + title + "}\n\\maketitle\n"
                        + "\\thispagestyle{empty}\n")
                elif plain == "Abstract":
                    output.append("\\begin{abstract}\n")
                    in_abstract = True
                else:
                    if in_abstract:
                        output.append("\\end{abstract}\n")
                        in_abstract = False
                    if plain.startswith("Appendix A.") or plain == "References":
                        output.append("\\clearpage\n")
                    command = {"h2": "section", "h3": "subsection", "h4": "subsubsection"}[token.tag]
                    output.append(f"\\{command}*{{{title}}}\n"
                        + f"\\addcontentsline{{toc}}{{{command}}}{{{title}}}\n")
                    if plain == "References":
                        output.append("\\begingroup\\small\\raggedright\n")
                        reference_group = True
                index += 3
            elif token.type == "paragraph_open":
                contents = tokens[index + 1]
                following = index + 3
                if contents.content.startswith(("**Authors:**", "**Affiliation:**", "**Email:**")):
                    index = following
                    continue
                if contents.content.startswith("**Keywords:**"):
                    if in_abstract:
                        output.append("\\end{abstract}\n")
                        in_abstract = False
                    keyword_text = contents.content.removeprefix("**Keywords:**").strip()
                    output.append("\\begin{keywords}\n" + escape(keyword_text)
                        + "\n\\end{keywords}\n")
                elif len(contents.children) == 1 and contents.children[0].type == "image":
                    caption = escape(contents.children[0].content)
                    if (following < len(tokens) and tokens[following].type == "paragraph_open"
                            and tokens[following + 1].content.startswith("**Figure ")):
                        caption = self.inline(tokens[following + 1].children)
                        following += 3
                    output.append(self.figure(contents.children[0], caption))
                elif (contents.content.startswith("**Table ") and following < len(tokens)
                        and tokens[following].type == "table_open"):
                    end = next(i for i in range(following, len(tokens)) if tokens[i].type == "table_close")
                    output.append(self.table(tokens[following:end + 1], self.inline(contents.children)))
                    following = end + 1
                else:
                    output.append(self.inline(contents.children) + "\n\n")
                index = following
            elif token.type == "table_open":
                end = next(i for i in range(index, len(tokens)) if tokens[i].type == "table_close")
                output.append(self.table(tokens[index:end + 1]))
                index = end + 1
            elif token.type == "math_block":
                output.append("\\[\n" + token.content + "\n\\]\n\n")
                self.equations += 1
                index += 1
            elif token.type == "fence":
                output.append("\\begin{Verbatim}[fontsize=\\small,breaklines=true,breakanywhere=true]\n"
                    + token.content + "\\end{Verbatim}\n")
                index += 1
            elif token.type in {"ordered_list_open", "bullet_list_open"}:
                environment = "enumerate" if token.type == "ordered_list_open" else "itemize"
                output.append("\\begin{" + environment + "}\n")
                index += 1
            elif token.type in {"ordered_list_close", "bullet_list_close"}:
                environment = "enumerate" if token.type == "ordered_list_close" else "itemize"
                output.append("\\end{" + environment + "}\n")
                index += 1
            elif token.type in {"list_item_open", "list_item_close"}:
                if token.type == "list_item_open":
                    output.append("\\item ")
                index += 1
            else:
                raise ValueError(f"Unsupported manuscript block: {token.type}")
        if reference_group:
            output.append("\\endgroup\n")
        if self.artifacts:
            if len(self.artifacts) > 6:
                output.append("\\clearpage\n")
            output.append("\\section*{Artifact paths in the companion bundle}\n"
                "Local evidence links in this PDF point to this index. Extract the companion "
                "materials archive and open the listed paths from its project root.\n")
            for path, number in self.artifacts.items():
                output.append(r"\par\hypertarget{artifact-" + str(number) + r"}{}\textbf{[A"
                    + str(number) + r"]} \path{" + path + "}\n")
        title = next(line[2:] for line in text.splitlines() if line.startswith("# "))
        preamble = PREAMBLE.replace("DOCUMENT_TITLE", escape(title))
        short_title = "Hegel Machine: Recursive Hypothesis Research"
        preamble = preamble.replace("DOCUMENT_SHORT_TITLE", escape(short_title))
        preamble = preamble.replace("from paper/manuscript.md.", "from paper/" + self.source.name + ".")
        return preamble + "\n".join(output) + "\n\\end{document}\n"


PREAMBLE = r"""% Generated by scripts/build_manuscript.py from paper/manuscript.md.
% Edit the Markdown and rebuild to keep the two sources synchronized.
\documentclass[twoside,11pt]{article}
% The renderer uses explicit caption formatting on the underlying article class.
\makeatletter
\let\articlemakecaption\@makecaption
\makeatother
\usepackage[abbrvbib,preprint]{jmlr_style/jmlr2e}
\usepackage{fontspec}
\setmainfont[SmallCapsFont={Latin Modern Roman Caps}]{Latin Modern Roman}
\setmonofont[Scale=MatchLowercase]{Latin Modern Mono}
\usepackage{amsmath}
\usepackage{unicode-math}
\setmathfont{Latin Modern Math}
\usepackage{graphicx,booktabs,longtable,array}
% Restore the article caption implementation before caption applies our format.
\makeatletter
\let\@makecaption\articlemakecaption
\makeatother
\usepackage{caption}
\captionsetup{font=small,justification=raggedright,singlelinecheck=false}
\usepackage{fvextra}
\usepackage{xurl}
\hypersetup{hypertexnames=false,pdftitle={DOCUMENT_TITLE},
  pdfauthor={Yifan Zhang and Liang Zheng}}
\urlstyle{same}
\setlength{\emergencystretch}{2em}
\setlength{\LTpre}{.7em}
\setlength{\LTpost}{.7em}
\setlength{\textfloatsep}{7pt plus 1pt minus 2pt}
\setlength{\floatsep}{7pt plus 1pt minus 2pt}
\setlength{\intextsep}{7pt plus 1pt minus 2pt}
\setlength{\abovecaptionskip}{4pt}
\setlength{\belowcaptionskip}{0pt}
\widowpenalty=10000
\clubpenalty=10000
\raggedbottom
\ShortHeadings{DOCUMENT_SHORT_TITLE}{Zhang and Zheng}
\firstpageno{1}
\author{\name Yifan Zhang \email erzhu419@gmail.com; 204201048@csu.edu.cn\\
\addr Central South University\\
Changsha, Hunan, China
\AND
\name Liang Zheng \email zhengliang@csu.edu.cn\\
\addr Central South University\\
Changsha, Hunan, China}
\date{}
\begin{document}
"""


def build(source, *, tex_only=False):
    manuscript = Manuscript(source)
    tex = source.with_suffix(".tex")
    content = source.read_text()
    generated_tables = {}
    if source.name == "manuscript.md":
        generated_tables = generate_tables(source.parent.parent)
        for number, table in generated_tables.items():
            pattern = rf"(^\*\*Table {number}\.[^\n]*\n\n)(?:\|[^\n]*\|\n)+"
            content, count = re.subn(pattern, lambda match: match[1] + table + "\n", content,
                                     flags=re.MULTILINE)
            if count != 1:
                raise ValueError(f"Expected one manuscript Table {number}, found {count}")
        if content != source.read_text():
            source.write_text(content, encoding="utf-8")
    tex.write_text(manuscript.render(content), encoding="utf-8")
    report = {"source": str(source), "tex": str(tex), "figures": manuscript.figures,
              "local_artifacts": manuscript.artifacts,
              "tables": manuscript.tables, "display_equations": manuscript.equations,
              "tables_regenerated_from_results": list(generated_tables)}
    if tex_only:
        return report
    build_dir = source.parent / ".latex-build"
    build_dir.mkdir(exist_ok=True)
    command = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
               "-output-directory=" + str(build_dir), tex.name]
    for pass_number in (1, 2):
        completed = subprocess.run(command, cwd=source.parent, capture_output=True, text=True)
        (build_dir / f"{source.stem}-pass-{pass_number}.txt").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"XeLaTeX failed: {build_dir / f'{source.stem}-pass-{pass_number}.txt'}\n"
                               + completed.stdout[-4000:])
    pdf = source.with_suffix(".pdf")
    shutil.copyfile(build_dir / pdf.name, pdf)
    log = (build_dir / (tex.stem + ".log")).read_text(errors="replace")
    report.update(pdf=str(pdf), build_log=str(build_dir / (tex.stem + ".log")),
        overfull_boxes=re.findall(r"Overfull \\[hv]box[^\n]*", log),
        missing_characters=re.findall(r"Missing character[^\n]*", log),
        latex_warnings=[line for line in log.splitlines() if "Warning:" in line])
    import fitz
    with fitz.open(pdf) as document:
        report["pages"] = len(document)
        # Physical-page overflow is a rendering defect, distinct from narrow
        # column warnings in the TeX log. Page-number blocks are permitted.
        report["page_overflow"] = [{"page": index + 1, "text": block[4][:100]}
            for index, page in enumerate(document) for block in page.get_text("blocks")
            if block[6] == 0 and (block[0] < 0 or block[1] < 0
                or block[2] > page.rect.width + 1 or block[3] > page.rect.height + 1)]
    (build_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (build_dir / f"{source.stem}_build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--source", type=Path, default=ROOT / "paper/manuscript.md")
    arguments.add_argument("--tex-only", action="store_true")
    options = arguments.parse_args()
    print(json.dumps(build(options.source.resolve(), tex_only=options.tex_only), indent=2))


if __name__ == "__main__":
    main()
