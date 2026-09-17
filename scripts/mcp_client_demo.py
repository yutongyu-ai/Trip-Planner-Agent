"""Demo MCP client: connects to app/mcp_server.py over stdio, lists its
tools, and calls search_attractions once against the real Google Places
API (needs GOOGLE_PLACES_API_KEY in the repo-root .env, same as the main
app -- this makes one real, billable-tier API call).

Run from the repo root: python scripts/mcp_client_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"


async def main() -> None:
    # Places can return non-Latin place names/descriptions; Windows consoles
    # often default to a codepage (e.g. gbk) that can't print them.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=str(BACKEND_DIR),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools exposed by app/mcp_server.py:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description.splitlines()[0]}")

            print("\nCalling search_attractions(city='Porto', interests=['museums'], max_results=3)...")
            result = await session.call_tool(
                "search_attractions", {"city": "Porto", "interests": ["museums"], "max_results": 3}
            )
            for block in result.content:
                print(block.text if hasattr(block, "text") else block)


if __name__ == "__main__":
    asyncio.run(main())
