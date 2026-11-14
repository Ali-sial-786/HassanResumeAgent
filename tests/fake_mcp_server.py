"""Tiny stdio MCP server standing in for @playwright/mcp in tests."""
from mcp.server.mcpserver import MCPServer

server = MCPServer("fake-browser")


@server.tool()
def browser_navigate(url: str) -> str:
    """Navigate to a URL."""
    return f"navigated to {url}"


@server.tool()
def browser_snapshot() -> str:
    """Return a page snapshot."""
    return "- heading 'Apply for Engineer'\n- button 'Submit' [ref=e1]"


if __name__ == "__main__":
    server.run()
