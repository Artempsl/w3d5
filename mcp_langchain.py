"""
Document Analysis using MCP-Enabled Agent

This script analyzes business documents using the MCP filesystem agent.
"""

import sys
import asyncio
import logging
from pathlib import Path

from mcp_agent import MCPAgent
from config import Config, setup_logging

logger = logging.getLogger(__name__)


# Document analysis task prompt
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

6. Save the complete analysis to a new file called "consolidated_report.txt" using the write_file tool.

Make sure to use the filesystem tools to read actual file contents rather than guessing. Include specific references to the original documents where relevant.

Begin your analysis now.
"""


async def run_document_analysis():
    """Execute document analysis task with MCP agent."""
    
    logger.info("="*70)
    logger.info("DOCUMENT ANALYSIS TASK - MCP APPROACH")
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
    agent = MCPAgent()
    
    try:
        logger.info("Initializing MCP Agent...")
        await agent.initialize()
        
        logger.info("Connecting to MCP server and loading tools...")
        async with agent.mcp_manager.connect():
            tools = await agent.mcp_manager.get_langchain_tools()
            agent.agent = await agent.create_agent(tools)
            
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
            output_file = Path(Config.DOCUMENTS_DIR, "consolidated_report.txt")
            if output_file.exists():
                logger.info("")
                logger.info("✓ consolidated_report.txt created successfully")
                logger.info(f"  File size: {output_file.stat().st_size} bytes")
                logger.info(f"  Location: {output_file}")
            else:
                logger.warning("⚠ consolidated_report.txt was not created")
            
            logger.info("")
            logger.info("="*70)
            logger.info("✓ DOCUMENT ANALYSIS TASK COMPLETED")
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
