"""
Test script for MCP Filesystem Server validation.

Tests tool registration and basic functionality without running the full server.
"""

import sys
import asyncio
import logging
from pathlib import Path
from config import Config, setup_logging
from mcp_server import FilesystemMCPServer

async def test_mcp_server():
    """Test MCP server initialization and tool registration."""
    
    logger = logging.getLogger(__name__)
    logger.info("Testing MCP Filesystem Server...")
    
    # Initialize server
    try:
        server = FilesystemMCPServer(Config.MCP_ALLOWED_DIRECTORY)
        logger.info("✓ Server initialized successfully")
    except Exception as e:
        logger.error(f"✗ Server initialization failed: {e}")
        return False
    
    # Test path validation
    try:
        # Valid path (within allowed directory)
        test_path = server._validate_path("financial_risks.txt")
        logger.info(f"✓ Path validation working: {test_path.name}")
        
        # Invalid path (outside allowed directory)
        try:
            server._validate_path("C:\\Windows\\System32\\test.txt")
            logger.error("✗ Security check failed - allowed access outside directory")
            return False
        except ValueError:
            logger.info("✓ Security check passed - blocked outside access")
    
    except Exception as e:
        logger.error(f"✗ Path validation failed: {e}")
        return False
    
    # Test tool methods (direct invocation)
    try:
        # Test list_directory
        result = await server._list_directory(".")
        logger.info(f"✓ list_directory working: {len(result)} result(s)")
        
        # Test read_file (on an existing file)
        test_files = ["financial_risks.txt", "marketing_strategy.txt", "sales_performance.txt"]
        for test_file in test_files:
            if Path(Config.DOCUMENTS_DIR, test_file).exists():
                result = await server._read_file(test_file)
                logger.info(f"✓ read_file working: read {test_file}")
                break
        else:
            logger.warning("⚠ No test documents found for read_file test")
        
        # Test write_file
        test_content = "MCP Server Test - " + str(asyncio.get_event_loop().time())
        result = await server._write_file("mcp_test_output.txt", test_content)
        logger.info(f"✓ write_file working: wrote test file")
        
        # Verify file was written
        test_file_path = Path(Config.DOCUMENTS_DIR, "mcp_test_output.txt")
        if test_file_path.exists():
            content = test_file_path.read_text()
            if content == test_content:
                logger.info("✓ File write verification passed")
            else:
                logger.error("✗ File content mismatch")
                return False
        
    except Exception as e:
        logger.error(f"✗ Tool method test failed: {e}")
        return False
    
    logger.info("="*60)
    logger.info("✓ ALL MCP SERVER TESTS PASSED")
    logger.info("="*60)
    return True


if __name__ == "__main__":
    setup_logging("INFO")
    
    success = asyncio.run(test_mcp_server())
    
    sys.exit(0 if success else 1)
