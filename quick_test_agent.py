"""Quick test of MCP agent."""
import asyncio
import logging
from mcp_agent import MCPAgent
from config import setup_logging

async def quick_test():
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    agent = MCPAgent()
    
    await agent.initialize()
    
    async with agent.mcp_manager.connect():
        tools = await agent.mcp_manager.get_langchain_tools()
        agent.agent = await agent.create_agent(tools)
        
        # Simple query
        result = await agent.run("What is 2+2?")
        
        print("\n" + "="*60)
        print("RESULT:")
        print(result.get("output", "No output"))
        print("="*60)

asyncio.run(quick_test())
