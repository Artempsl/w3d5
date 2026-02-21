"""
MCP Client Connection Module

Connects to MCP filesystem server and converts tools to LangChain format.
"""

import sys
import asyncio
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from config import Config

logger = logging.getLogger(__name__)


class MCPClientManager:
    """
    Manages MCP client connection and tool conversion.
    
    Provides a clean interface for connecting to MCP server and
    retrieving LangChain-compatible tools.
    """
    
    def __init__(self):
        """Initialize MCP client manager."""
        self.session: Optional[ClientSession] = None
        self.tools: List = []
        self._cleanup_handlers = []
    
    @asynccontextmanager
    async def connect(self):
        """
        Connect to MCP server via stdio.
        
        Usage:
            async with client_manager.connect() as session:
                # Use session
                tools = await session.list_tools()
        """
        # Configure server parameters
        server_params = StdioServerParameters(
            command=sys.executable,  # Use current Python interpreter
            args=["mcp_server.py"],  # Run our MCP server
            env=None  # Inherit environment
        )
        
        logger.info(f"Connecting to MCP server: {server_params.command} {' '.join(server_params.args)}")
        
        # Create stdio client connection
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize session
                await session.initialize()
                
                logger.info("✓ MCP client connected and initialized")
                self.session = session
                
                try:
                    yield session
                finally:
                    self.session = None
                    logger.info("MCP client disconnected")
    
    async def list_tools(self) -> List[dict]:
        """
        List all available tools from MCP server.
        
        Returns:
            List of tool definitions with name, description, and schema
        """
        if not self.session:
            raise RuntimeError("Not connected. Use 'async with connect()' first.")
        
        response = await self.session.list_tools()
        
        tools_info = []
        for tool in response.tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            })
        
        logger.info(f"Discovered {len(tools_info)} MCP tools")
        
        return tools_info
    
    async def get_langchain_tools(self):
        """
        Convert MCP tools to LangChain format.
        
        Returns:
            List of LangChain-compatible tool objects
        """
        if not self.session:
            raise RuntimeError("Not connected. Use 'async with connect()' first.")
        
        # Use langchain-mcp-adapters for automatic conversion
        logger.info("Converting MCP tools to LangChain format...")
        
        langchain_tools = await load_mcp_tools(self.session)
        
        logger.info(f"✓ Converted {len(langchain_tools)} tools to LangChain format")
        
        for tool in langchain_tools:
            logger.info(f"  - {tool.name}: {tool.description}")
        
        self.tools = langchain_tools
        return langchain_tools


async def test_mcp_connection():
    """Test MCP client connection and tool discovery."""
    
    logger.info("Testing MCP Client Connection...")
    logger.info("="*60)
    
    manager = MCPClientManager()
    
    try:
        async with manager.connect() as session:
            # List available tools
            tools = await manager.list_tools()
            
            logger.info("\nDiscovered MCP Tools:")
            logger.info("-"*60)
            for i, tool in enumerate(tools, 1):
                logger.info(f"{i}. {tool['name']}")
                logger.info(f"   Description: {tool['description']}")
                logger.info(f"   Required params: {tool['input_schema'].get('required', [])}")
            
            # Convert to LangChain format
            logger.info("\n" + "-"*60)
            langchain_tools = await manager.get_langchain_tools()
            
            logger.info("\nLangChain Tools:")
            logger.info("-"*60)
            for i, tool in enumerate(langchain_tools, 1):
                logger.info(f"{i}. {tool.name}")
                logger.info(f"   Type: {type(tool).__name__}")
                logger.info(f"   Description: {tool.description[:80]}...")
            
            # Test a tool invocation
            logger.info("\n" + "-"*60)
            logger.info("Testing tool invocation: list_directory")
            
            # Find the list_directory tool
            list_dir_tool = next(
                (t for t in langchain_tools if t.name == "list_directory"),
                None
            )
            
            if list_dir_tool:
                result = await list_dir_tool.ainvoke({"path": "."})
                logger.info(f"✓ Tool invocation successful")
                logger.info(f"Result preview: {result[:200]}...")
            else:
                logger.warning("list_directory tool not found")
            
            logger.info("\n" + "="*60)
            logger.info("✓ MCP CLIENT CONNECTION TEST PASSED")
            logger.info("="*60)
            
            return True
    
    except Exception as e:
        logger.error(f"✗ MCP connection test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    from config import setup_logging
    
    setup_logging("INFO")
    
    success = asyncio.run(test_mcp_connection())
    
    sys.exit(0 if success else 1)
