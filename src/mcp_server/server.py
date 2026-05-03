"""MCP server entry point — stdio transport for Claude Desktop compatibility."""

from __future__ import annotations

import asyncio
from src.core.logging import get_logger

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from src.mcp_server.tools import TOOLS, dispatch

logger = get_logger(__name__)


async def _run() -> None:
    load_dotenv()

    from src.db import connection as db
    await db.run_migrations()

    server = Server("documind")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        logger.info("MCP tool call: %s args=%s", name, list(arguments.keys()))
        return await dispatch(name, arguments)

    logger.info("DocuMind MCP server starting (stdio transport)")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """CLI entry point registered in pyproject.toml."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
