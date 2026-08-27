"""发送给支持原生 Function Calling 模型的 JSON Schema。"""

from __future__ import annotations

FILE_TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出工作区内的文件和目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "工作区相对目录。"},
                    "max_depth": {"type": "integer", "minimum": 1, "maximum": 6},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取 UTF-8 文本文件并返回行号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "在工作区文本文件中搜索字面字符串。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "创建 UTF-8 文本文件；仅在明确允许时覆盖。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "在一个 UTF-8 文件中精确替换文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "expected_replacements": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_DEFINITIONS = list(FILE_TOOL_DEFINITIONS)

