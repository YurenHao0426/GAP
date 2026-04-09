#!/usr/bin/env python3
"""
Streamlit-based PutnamGAP dataset viewer.

Features:
- Scans preprocess/PutnamGAP for JSON files and allows prev/next navigation
- Select specific file from a dropdown
- Choose which variant to display: original or one of:
  descriptive_long, descriptive_long_confusing, descriptive_long_misleading, garbled_string, kernel_variant
- Toggle to show Question, Solution (a.k.a. Answer), or Both
- TeX rendering via Markdown by default, with optional HTML+MathJax fallback
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from streamlit.components.v1 import html as st_html


DATA_DIR = os.path.join(os.path.dirname(__file__), "PutnamGAP")
SUPPORTED_VARIANTS = [
    "original",
    "descriptive_long",
    "descriptive_long_confusing",
    "descriptive_long_misleading",
    "garbled_string",
    "kernel_variant",
]


def discover_json_files(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".json")
    ]
    files.sort()
    return files


def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_original_qa(d: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    # Prefer "question"/"solution"; gracefully fall back to "answer" if present
    question: Optional[str] = d.get("question")
    solution: Optional[str] = d.get("solution", d.get("answer"))
    return question, solution


def get_variant_qa(
    d: Dict[str, Any], variant_key: str
) -> Tuple[Optional[str], Optional[str]]:
    variants = d.get("variants")
    if not isinstance(variants, dict):
        return None, None
    var = variants.get(variant_key)
    if not isinstance(var, dict):
        return None, None
    question: Optional[str] = var.get("question")
    solution: Optional[str] = var.get("solution", var.get("answer"))
    return question, solution


def render_markdown_with_math(text: str) -> None:
    # Streamlit markdown supports MathJax ($...$, $$...$$)
    st.markdown(text, unsafe_allow_html=True)


def render_with_mathjax_html(blocks: List[Tuple[str, str]]) -> None:
    """
    Render content with MathJax v3 inside a single HTML component.
    blocks: list of (heading, content) tuples
    """
    # Build a small HTML page with MathJax v3; render all blocks together.
    content_sections = []
    for heading, content in blocks:
        section_html = f"""
        <section style="margin-bottom: 1.25rem;">
          <h3 style="margin: 0 0 .5rem 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;">
            {heading}
          </h3>
          <div class="mj-content">{content}</div>
        </section>
        """
        content_sections.append(section_html)

    page = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script>
      window.MathJax = {{
        tex: {{
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
        }},
        svg: {{ fontCache: 'global' }}
      }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
    <style>
      html, body {{
        background: #0f0f10;
        color: #f5f6f7;
      }}
      body {{
        padding: 0.5rem 0.25rem;
        color: #f5f6f7;
        background: #0f0f10;
      }}
      .mj-content {{
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        font-size: 1rem;
        color: #f5f6f7;
        background: #0f0f10;
        padding: 0.25rem 0.25rem;
        border-radius: 4px;
      }}
      code, pre {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        color: #e6e6e6;
      }}
      svg, .MathJax, .mjx-svg, .mjx-mrow {{
        color: #f5f6f7;
      }}
    </style>
  </head>
  <body>
    {''.join(content_sections)}
  </body>
</html>
"""
    st_html(page, height=600, scrolling=True)


def main() -> None:
    st.set_page_config(page_title="PutnamGAP Viewer", layout="wide")
    st.title("PutnamGAP 数据可视化与校对")
    st.caption("浏览原题与不同变体；支持 TeX 渲染与文件前后切换。")

    files = discover_json_files(DATA_DIR)
    if not files:
        st.error(f"未在目录中发现 JSON 文件：{DATA_DIR}")
        st.stop()

    # Sidebar controls
    with st.sidebar:
        st.subheader("文件与显示设置")

        # Single source of truth for navigation: file_index
        file_labels = [os.path.basename(p) for p in files]
        if "file_index" not in st.session_state:
            st.session_state.file_index = 0

        selected_label = st.selectbox(
            "选择题目文件",
            options=file_labels,
            index=st.session_state.file_index,
        )
        # Sync file_index if user chose a different label
        current_index = file_labels.index(selected_label)
        if current_index != st.session_state.file_index:
            st.session_state.file_index = current_index

        # Variant selection
        variant_human_labels = {
            "original": "原题 original",
            "descriptive_long": "descriptive_long",
            "descriptive_long_confusing": "descriptive_long_confusing",
            "descriptive_long_misleading": "descriptive_long_misleading",
            "garbled_string": "garbled_string",
            "kernel_variant": "kernel_variant",
        }
        variant_choice_label = st.radio(
            "选择显示内容",
            options=[variant_human_labels[k] for k in SUPPORTED_VARIANTS],
            index=0,
        )
        # Reverse map to internal key
        selected_variant = {
            v: k for k, v in variant_human_labels.items()
        }[variant_choice_label]

        # Display options
        show_mode = st.radio(
            "显示部分",
            options=["Question", "Solution", "Both"],
            index=0,
            horizontal=True,
        )
        render_mode = st.radio(
            "渲染方式",
            options=["Markdown (默认)", "HTML + MathJax"],
            index=1,
        )
        show_meta = st.checkbox("显示原始 JSON 片段", value=False)

    # Prev/Next navigation buttons in header row
    left, mid, right = st.columns([1, 6, 1])
    with left:
        if st.button("⬅️ 上一题", use_container_width=True):
            new_index = (st.session_state.file_index - 1) % len(files)
            st.session_state.file_index = new_index
            st.rerun()
    with right:
        if st.button("下一题 ➡️", use_container_width=True):
            new_index = (st.session_state.file_index + 1) % len(files)
            st.session_state.file_index = new_index
            st.rerun()

    current_path = files[st.session_state.file_index]
    data = load_json(current_path)

    st.write(f"当前文件：`{os.path.basename(current_path)}`  （{st.session_state.file_index + 1}/{len(files)}）")
    st.divider()

    # Resolve question/solution for chosen variant
    if selected_variant == "original":
        q_text, s_text = get_original_qa(data)
    else:
        q_text, s_text = get_variant_qa(data, selected_variant)

    # Assemble content blocks to render
    blocks: List[Tuple[str, str]] = []
    if show_mode in ("Question", "Both"):
        if q_text:
            blocks.append(("Question", q_text))
        else:
            st.warning("该选择下未找到 Question。")
    if show_mode in ("Solution", "Both"):
        if s_text:
            blocks.append(("Solution", s_text))
        else:
            st.warning("该选择下未找到 Solution/Answer。")

    if len(blocks) > 0:
        if render_mode.startswith("Markdown"):
            for heading, content in blocks:
                st.subheader(heading)
                render_markdown_with_math(content)
                st.markdown("---")
        else:
            render_with_mathjax_html(blocks)
    else:
        st.info("无可显示内容。")

    if show_meta:
        with st.expander("原始 JSON（截断显示）", expanded=False):
            # Show a trimmed version to avoid overwhelming the UI
            preview: Dict[str, Any] = {}
            for k in ("index", "type", "tag", "difficulty", "problem_type"):
                if k in data:
                    preview[k] = data[k]
            preview["keys"] = list(data.keys())
            st.json(preview)

            st.caption("完整 JSON 路径：")
            st.code(current_path)

    st.caption("提示：可以使用侧边栏选择具体文件与变体，也可通过顶部按钮快速前后切换 JSON。")


if __name__ == "__main__":
    main()


