"""
MCP-Enabled LangChain Agent

Creates a ReAct agent with MCP filesystem tools for document analysis.
"""

import sys
import asyncio
import logging
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from config import Config, setup_logging
from mcp_client import MCPClientManager

logger = logging.getLogger(__name__)


# System prompt for agent
SYSTEM_PROMPT = """You are an AI assistant with access to filesystem tools via MCP (Model Context Protocol).

IMPORTANT GUIDELINES:
- Always use tools to read actual file contents - never guess or make up content
- When analyzing multiple documents, read each one completely
- Be thorough and systematic in your analysis
- Provide specific evidence from documents when making claims
- If you need to create a report, use the write_file tool
- Think step by step and explain your reasoning

Available tools allow you to:
- read_file: Read the complete contents of text files
- list_directory: List all files and directories in a path
- write_file: Create or overwrite files with new content
"""


class MCPAgent:
    """
    LangChain agent with MCP tools for document analysis.
    
    Provides a clean interface for creating and running agents
    with MCP filesystem capabilities.
    """
    
    def __init__(self):
        """Initialize the MCP agent."""
        self.llm: Optional[ChatOpenAI] = None
        self.agent = None
        self.mcp_manager: Optional[MCPClientManager] = None
        self.tools = []
    
    async def initialize(self) -> None:
        """Initialize LLM and connect to MCP server."""
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {Config.LLM_MODEL}")
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            api_key=Config.OPENAI_API_KEY,
            max_tokens=Config.LLM_MAX_TOKENS
        )
        logger.info("✓ LLM initialized")
        
        # Connect to MCP server
        logger.info("Connecting to MCP server...")
        self.mcp_manager = MCPClientManager()
        
        # Note: MCP manager uses async context manager
        # Tools will be loaded when we enter the context
        logger.info("✓ MCP manager ready")
    
    async def create_agent(self, tools):
        """
        Create ReAct agent with given tools.
        
        Args:
            tools: List of LangChain tools
            
        Returns:
            LangGraph agent ready to run queries
        """
        logger.info(f"Creating ReAct agent with {len(tools)} tools...")
        
        # Create ReAct agent using LangChain
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT
        )
        
        logger.info("✓ Agent created successfully")
        
        return agent
    
    async def run(self, query: str) -> Dict[str, Any]:
        """
        Run agent with a query.
        
        Args:
            query: Question or task for the agent
            
        Returns:
            Dictionary with 'output' and execution metadata
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call setup() first.")
        
        logger.info(f"Running agent with query: {query[:100]}...")
        
        try:
            # Run agent
            final_result = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=query)]}
            )
            
            # Extract the final message
            final_message = final_result["messages"][-1].content
            
            logger.info("✓ Agent execution completed")
            return {"output": final_message, "messages": final_result["messages"]}
        
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise
    
    async def setup(self):
        """Setup agent with MCP tools (used with async context manager)."""
        await self.initialize()
        
        # Connect to MCP and get tools
        async with self.mcp_manager.connect():
            self.tools = await self.mcp_manager.get_langchain_tools()
            self.agent = await self.create_agent(self.tools)
            
            # Yield control back to caller
            yield self


async def test_agent():
    """Test agent with a simple query."""
    
    logger.info("="*60)
    logger.info("TESTING MCP-ENABLED LANGCHAIN AGENT")
    logger.info("="*60)
    
    agent = MCPAgent()
    
    # Simple test query
    test_query = "List all files in the current directory and tell me what document files you see."
    
    logger.info(f"\nTest Query: {test_query}\n")
    logger.info("-"*60)
    
    try:
        await agent.initialize()
        
        async with agent.mcp_manager.connect():
            tools = await agent.mcp_manager.get_langchain_tools()
            agent.agent = await agent.create_agent(tools)
            
            # Run test query
            print("\n" + "-"*60)
            result = await agent.run(test_query)
            
            logger.info("\n" + "="*60)
            logger.info("AGENT RESPONSE")
            logger.info("="*60)
            logger.info(result.get("output", "No output"))
            logger.info("="*60)
            
            logger.info("\n✓ AGENT TEST PASSED")
            return True
    
    except Exception as e:
        logger.error(f"\n✗ AGENT TEST FAILED: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    setup_logging("INFO")
    
    success = asyncio.run(test_agent())
    
    sys.exit(0 if success else 1)
