"""
MCP Filesystem Server - Pure Python Implementation

Provides secure filesystem access via Model Context Protocol.
All operations are scoped to the allowed directory for security.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource, ImageContent
import mcp.server.stdio

logger = logging.getLogger(__name__)


class FilesystemMCPServer:
    """
    MCP Server with filesystem tools.
    
    Security: All file operations are validated against allowed_directory.
    """
    
    def __init__(self, allowed_directory: str):
        """
        Initialize filesystem MCP server.
        
        Args:
            allowed_directory: Root directory for all file operations (security boundary)
        """
        self.allowed_directory = Path(allowed_directory).resolve()
        self.server = Server("filesystem-server")
        
        # Register tools
        self._register_tools()
        
        logger.info(f"MCP Filesystem Server initialized")
        logger.info(f"Allowed directory: {self.allowed_directory}")
    
    def _validate_path(self, file_path: str) -> Path:
        """
        Validate that path is within allowed directory.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Resolved absolute path
            
        Raises:
            ValueError: If path is outside allowed directory
        """
        # Convert to absolute path
        if os.path.isabs(file_path):
            full_path = Path(file_path).resolve()
        else:
            full_path = (self.allowed_directory / file_path).resolve()
        
        # Security check: ensure path is within allowed directory
        try:
            full_path.relative_to(self.allowed_directory)
        except ValueError:
            raise ValueError(
                f"Access denied: {file_path} is outside allowed directory "
                f"{self.allowed_directory}"
            )
        
        return full_path
    
    def _register_tools(self) -> None:
        """Register all filesystem tools with the MCP server."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available filesystem tools."""
            return [
                Tool(
                    name="read_file",
                    description="Read the complete contents of a text file. Returns file content as string.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read (relative or absolute)"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="list_directory",
                    description="List all files and directories in a given directory path.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to list (relative or absolute). Use '.' for current directory."
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="write_file",
                    description="Write content to a file. Creates file if it doesn't exist, overwrites if it does.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path where to write the file"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["path", "content"]
                    }
                ),
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Execute a tool by name with given arguments."""
            
            try:
                if name == "read_file":
                    return await self._read_file(arguments["path"])
                elif name == "list_directory":
                    return await self._list_directory(arguments["path"])
                elif name == "write_file":
                    return await self._write_file(arguments["path"], arguments["content"])
                else:
                    raise ValueError(f"Unknown tool: {name}")
            
            except Exception as e:
                logger.error(f"Tool execution failed: {name} - {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _read_file(self, path: str) -> List[TextContent]:
        """
        Read file contents.
        
        Args:
            path: File path to read
            
        Returns:
            List containing TextContent with file contents
        """
        file_path = self._validate_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            logger.info(f"Read file: {file_path.name} ({len(content)} chars)")
            return [TextContent(
                type="text",
                text=content
            )]
        except UnicodeDecodeError:
            raise ValueError(f"File is not a text file: {path}")
    
    async def _list_directory(self, path: str) -> List[TextContent]:
        """
        List directory contents.
        
        Args:
            path: Directory path to list
            
        Returns:
            List containing TextContent with directory listing
        """
        dir_path = self._validate_path(path)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        
        # Get all items in directory
        items = []
        for item in sorted(dir_path.iterdir()):
            item_type = "DIR" if item.is_dir() else "FILE"
            size = item.stat().st_size if item.is_file() else 0
            items.append(f"{item_type:6} {size:>10} bytes  {item.name}")
        
        result = f"Contents of {dir_path}:\n" + "\n".join(items)
        logger.info(f"Listed directory: {dir_path.name} ({len(items)} items)")
        
        return [TextContent(type="text", text=result)]
    
    async def _write_file(self, path: str, content: str) -> List[TextContent]:
        """
        Write content to file.
        
        Args:
            path: File path to write
            content: Content to write
            
        Returns:
            List containing TextContent with success message
        """
        file_path = self._validate_path(path)
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        file_path.write_text(content, encoding='utf-8')
        
        logger.info(f"Wrote file: {file_path.name} ({len(content)} chars)")
        
        return [TextContent(
            type="text",
            text=f"Successfully wrote {len(content)} characters to {file_path.name}"
        )]
    
    async def run(self) -> None:
        """Run the MCP server on stdio."""
        logger.info("Starting MCP server on stdio...")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point for MCP server."""
    import sys
    from config import Config
    
    # Setup logging to stderr (stdout is reserved for JSON-RPC protocol)
    logging.basicConfig(
        level=logging.ERROR,  # Only log errors to avoid interfering with protocol
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr  # Use stderr for logs, stdout for JSON-RPC
    )
    
    # Initialize server with allowed directory
    server = FilesystemMCPServer(Config.MCP_ALLOWED_DIRECTORY)
    
    # Run server
    await server.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
