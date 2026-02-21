"""
Direct Approach - Document Analysis WITHOUT MCP

This script performs the same document analysis as the MCP approach,
but uses direct file I/O with custom LangChain tools instead of MCP.
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from config import Config, setup_logging

logger = logging.getLogger(__name__)


# System prompt for agent
SYSTEM_PROMPT = """You are an AI assistant with access to filesystem tools.

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


# Custom filesystem tools (Direct approach - no MCP)
@tool
def read_file(path: str) -> str:
    """
    Read the complete contents of a text file. Returns file content as string.
    
    Args:
        path: Path to the file to read (relative or absolute)
    
    Returns:
        File contents as a string
    """
    try:
        # Resolve path
        if Path(path).is_absolute():
            file_path = Path(path)
        else:
            file_path = Path(Config.DOCUMENTS_DIR) / path
        
        # Security: ensure within allowed directory
        file_path = file_path.resolve()
        allowed_dir = Path(Config.DOCUMENTS_DIR).resolve()
        
        try:
            file_path.relative_to(allowed_dir)
        except ValueError:
            return f"Error: Access denied - path outside allowed directory"
        
        # Read file
        if not file_path.exists():
            return f"Error: File not found: {path}"
        
        if not file_path.is_file():
            return f"Error: Not a file: {path}"
        
        content = file_path.read_text(encoding='utf-8')
        logger.info(f"Read file: {file_path.name} ({len(content)} chars)")
        
        return content
    
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def list_directory(path: str) -> str:
    """
    List all files and directories in a given directory path.
    
    Args:
        path: Directory path to list (relative or absolute). Use '.' for current directory.
    
    Returns:
        Directory listing as a formatted string
    """
    try:
        # Resolve path
        if path == '.':
            dir_path = Path(Config.DOCUMENTS_DIR)
        elif Path(path).is_absolute():
            dir_path = Path(path)
        else:
            dir_path = Path(Config.DOCUMENTS_DIR) / path
        
        # Security: ensure within allowed directory
        dir_path = dir_path.resolve()
        allowed_dir = Path(Config.DOCUMENTS_DIR).resolve()
        
        try:
            dir_path.relative_to(allowed_dir)
        except ValueError:
            return f"Error: Access denied - path outside allowed directory"
        
        # List directory
        if not dir_path.exists():
            return f"Error: Directory not found: {path}"
        
        if not dir_path.is_dir():
            return f"Error: Not a directory: {path}"
        
        items = []
        for item in sorted(dir_path.iterdir()):
            item_type = "DIR" if item.is_dir() else "FILE"
            size = item.stat().st_size if item.is_file() else 0
            items.append(f"{item_type:6} {size:>10} bytes  {item.name}")
        
        result = f"Contents of {dir_path}:\n" + "\n".join(items)
        logger.info(f"Listed directory: {dir_path.name} ({len(items)} items)")
        
        return result
    
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a file. Creates file if it doesn't exist, overwrites if it does.
    
    Args:
        path: Path where to write the file
        content: Content to write to the file
    
    Returns:
        Success message
    """
    try:
        # Resolve path
        if Path(path).is_absolute():
            file_path = Path(path)
        else:
            file_path = Path(Config.DOCUMENTS_DIR) / path
        
        # Security: ensure within allowed directory
        file_path = file_path.resolve()
        allowed_dir = Path(Config.DOCUMENTS_DIR).resolve()
        
        try:
            file_path.relative_to(allowed_dir)
        except ValueError:
            return f"Error: Access denied - path outside allowed directory"
        
        # Write file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        
        logger.info(f"Wrote file: {file_path.name} ({len(content)} chars)")
        
        return f"Successfully wrote {len(content)} characters to {file_path.name}"
    
    except Exception as e:
        return f"Error writing file: {str(e)}"


class DirectAgent:
    """
    LangChain agent with direct filesystem tools (no MCP).
    """
    
    def __init__(self):
        """Initialize the direct agent."""
        self.llm: Optional[ChatOpenAI] = None
        self.agent = None
        self.tools = []
    
    async def initialize(self) -> None:
        """Initialize LLM and tools."""
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {Config.LLM_MODEL}")
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            api_key=Config.OPENAI_API_KEY,
            max_tokens=Config.LLM_MAX_TOKENS
        )
        logger.info("✓ LLM initialized")
        
        # Create tools list
        self.tools = [read_file, list_directory, write_file]
        logger.info(f"✓ {len(self.tools)} direct tools created")
    
    async def create_agent(self):
        """Create agent with direct tools."""
        logger.info(f"Creating agent with {len(self.tools)} tools...")
        
        # Create agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT
        )
        
        logger.info("✓ Agent created successfully")
    
    async def run(self, query: str) -> Dict[str, Any]:
        """
        Run agent with a query.
        
        Args:
            query: Question or task for the agent
            
        Returns:
            Dictionary with 'output' and execution metadata
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() and create_agent() first.")
        
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


# Document analysis task prompt (same as MCP approach)
ANALYSIS_TASK = """You are an AI assistant with access to filesystem tools.

Your task is to analyze the three documents in the current directory:
- "financial_risks.txt"
- "marketing_strategy.txt"
- "sales_performance.txt"

Please perform the following actions step by step:

1. List all files in the current directory to verify the documents exist.

2. Read each of the three documents completely:
   - financial_risks.txt
   - marketing_strategy.txt
   - sales_performance.txt

3. For each document, create a summary with 5-7 key bullet points highlighting:
   - Main themes and objectives
   - Critical data points or metrics
   - Major concerns or opportunities
   - Strategic recommendations

4. Identify any inconsistencies, risks, or conflicts between the three perspectives (marketing, sales, and financial).

5. Create a consolidated executive summary that:
   - Integrates insights from all three files
   - Highlights alignment and conflicts
   - Provides actionable recommendations
   - References specific data from the original documents

6. Save the complete analysis to a new file called "consolidated_report_direct.txt" using the write_file tool.

Make sure to use the filesystem tools to read actual file contents rather than guessing. Include specific references to the original documents where relevant.

Begin your analysis now.
"""


async def run_document_analysis():
    """Execute document analysis task with direct approach."""
    
    logger.info("="*70)
    logger.info("DOCUMENT ANALYSIS TASK - DIRECT APPROACH (NO MCP)")
    logger.info("="*70)
    
    # Check document files exist
    doc_files = [
        "financial_risks.txt",
        "marketing_strategy.txt",
        "sales_performance.txt"
    ]
    
    missing_files = []
    for doc in doc_files:
        if not Path(Config.DOCUMENTS_DIR, doc).exists():
            missing_files.append(doc)
    
    if missing_files:
        logger.error(f"Missing document files: {missing_files}")
        return False
    
    logger.info(f"✓ All {len(doc_files)} documents found")
    logger.info("")
    
    # Initialize agent
    agent = DirectAgent()
    
    try:
        logger.info("Initializing Direct Agent...")
        await agent.initialize()
        await agent.create_agent()
        
        logger.info("✓ Agent ready")
        logger.info("")
        logger.info("="*70)
        logger.info("STARTING DOCUMENT ANALYSIS")
        logger.info("="*70)
        logger.info("")
        
        # Run analysis
        result = await agent.run(ANALYSIS_TASK)
        
        logger.info("")
        logger.info("="*70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*70)
        logger.info("")
        
        # Display result
        output = result.get("output", "No output received")
        logger.info("Agent Response:")
        logger.info("-"*70)
        logger.info(output)
        logger.info("-"*70)
        
        # Verify output file was created
        output_file = Path(Config.DOCUMENTS_DIR, "consolidated_report_direct.txt")
        if output_file.exists():
            logger.info("")
            logger.info("✓ consolidated_report_direct.txt created successfully")
            logger.info(f"  File size: {output_file.stat().st_size} bytes")
            logger.info(f"  Location: {output_file}")
        else:
            logger.warning("⚠ consolidated_report_direct.txt was not created")
        
        logger.info("")
        logger.info("="*70)
        logger.info("✓ DOCUMENT ANALYSIS TASK COMPLETED (DIRECT APPROACH)")
        logger.info("="*70)
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Document analysis failed: {e}", exc_info=True)
        return False


async def main():
    """Main entry point."""
    setup_logging("INFO")
    
    success = await run_document_analysis()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
