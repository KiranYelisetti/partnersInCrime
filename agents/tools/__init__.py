"""
Tools package — export grouped tool lists for agent registration.
"""
from tools.file_tools import FILE_TOOLS, REFERENCE_TOOLS
from tools.git_tools import GIT_TOOLS
from tools.terminal_tools import TERMINAL_TOOLS
from tools.web_tools import WEB_TOOLS

# All tools combined (reference tools auto-included only when configured)
ALL_TOOLS = FILE_TOOLS + GIT_TOOLS + TERMINAL_TOOLS + WEB_TOOLS + REFERENCE_TOOLS

__all__ = [
    "FILE_TOOLS", "REFERENCE_TOOLS", "GIT_TOOLS",
    "TERMINAL_TOOLS", "WEB_TOOLS", "ALL_TOOLS",
]
