"""MCP server entry point — stdio transport for Claude Desktop compatibility."""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server

from src.mcp_server.tools import register_tools

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def _run() -> None:
    load_dotenv()

    from src.db import connection as db
    await db.run_migrations()

    server = Server("documind")
    register_tools(server)

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
